from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
import os
import json

app = Flask(__name__)

# Set up Gemini API key from environment variables
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

MIN_QUESTIONS = 3
MAX_QUESTIONS = 7

# Load sample prompts from JSON
with open("sample_prompts.json", "r") as file:
    sample_data = json.load(file)

# Global session storage
user_session = {
    "query": None,
    "responses": [],
    "questions": [],
    "question_count": 0,
    "scores": [],
}

def call_gemini(prompt, max_tokens=100, temperature=0.5):
    """Helper function to call Google's Gemini API."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip() if response and response.text else ""
    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return ""

def generate_question():
    """
    Generate a follow-up question dynamically based on full conversation history,
    ensuring logical progression without hallucinations.
    """
    if not user_session["responses"]:  # If no responses yet, provide a starting question
        return "What aspect of AI/ML interests you the most?"

    conversation_context = "\n".join([
        f"Q: {q}\nA: {a}" for q, a in zip(user_session["questions"], user_session["responses"])
    ])

    examples = sample_data["EXAMPLES"]

    # Format examples for Gemini (showing different levels)
    examples_text = ""
    for level, interactions in examples.items():
        examples_text += f"\n### {level} User Examples ###\n"
        for ex in interactions[:5]:  # Show 5 examples per level
            examples_text += (
                f"User was asked: \"{ex['follow_up_question']}\"\n"
                f"User responded: \"{ex['user_response']}\"\n"
                f"Expected follow-up: \"{ex['expected_follow_up']}\"\n\n"
            )

    # Instruction for Gemini
    prompt = (
        f"I am developing an AI chatbot that adapts to users at different AI/ML experience levels.\n"
        f"Users may be beginners, intermediates, or advanced practitioners.\n\n"
        f"Below is the conversation so far:\n\n"
        f"{conversation_context}\n\n"
        f"Here are some sample interactions from different levels:\n"
        f"{examples_text}\n"
        f"The latest user response is: \"{user_session['responses'][-1]}\"\n\n"
        f"Now, generate the next follow-up question that logically continues this conversation.\n"
        f"STRICT RULES:\n"
        f"- DO NOT change the topic or reset the discussion.\n"
        f"- If the user is vague or uncertain (e.g., 'I don't know' or 'I'm not sure'), clarify while staying within the SAME topic.\n"
        f"- Ensure the question is engaging, relevant, and builds upon the user's last clear statement.\n"
    )
    print(prompt)
    # Generate the next follow-up question using Gemini
    generated_question = call_gemini(prompt, max_tokens=50, temperature=0.7)

    # 🔹 If Gemini fails to generate a response, ask for clarification based on the last user response
    if not generated_question:
        return f"I didn't fully understand your last response: \"{user_session['responses'][-1]}\". Could you clarify what you meant?"

    return generated_question

def score_response(response, question):
    """
    Evaluate user response and infer expertise level along with whether more questions are needed.
    """
    prompt = (
        f"Evaluate this user response in relation to the follow-up question in AI/ML.\n\n"
        f"Follow-up question: \"{question}\"\n"
        f"User response: \"{response}\"\n\n"
        "On a scale from 0 to 5, where 0 means no relevant information and 5 means an exceptional response, "
        "score the response. Also, analyze if the response aligns more with a Beginner, Intermediate, or Advanced user."
        "Finally, decide if the chatbot should ask more follow-up questions (Yes/No).\n"
        "Provide output as:\n"
        "Final Score: X\n"
        "Reasoning: Explain why the response received this score.\n"
        "User Level: [Beginner/Intermediate/Advanced]\n"
        "More Questions Needed: [Yes/No]")

    result = call_gemini(prompt, max_tokens=150, temperature=0.3)

    score = 0
    reasoning = "No explanation provided."
    user_level = "Unknown"
    more_questions_needed = "Yes"

    try:
        for line in result.splitlines():
            if "Final Score:" in line:
                try:
                    score = int(line.split(":")[1].strip())
                except ValueError:
                    score = 0  # Default to 0 if conversion fails
            if "Reasoning:" in line:
                reasoning = line.split(":", 1)[1].strip()
            if "User Level:" in line:
                user_level = line.split(":")[1].strip()
            if "More Questions Needed:" in line:
                more_questions_needed = line.split(":")[1].strip()
    except Exception as e:
        print(f"Error extracting score: {e}")

    return score, reasoning, user_level, more_questions_needed

def generate_summary(responses):
    """Generate a summary of what the user wants to learn based on their responses."""
    prompt = (
        f"Summarize the user's learning goals based on these responses:\n"
        f"{responses}\n"
        "Provide a concise summary.")
    return call_gemini(prompt, max_tokens=100, temperature=0.5)

def extract_key_elements(summary, user_level):
    """Extract key elements from the summary and include the user's level."""
    prompt = (f"Extract the key topics from the following summary:\n"
              f"{summary}\n"
              f"Ensure to include the user's level: {user_level}.\n"
              "Provide only the key elements as a comma-separated list.")
    key_elements = call_gemini(prompt, max_tokens=50, temperature=0.3)

    elements = key_elements.split(", ")
    if user_level.lower() not in [e.lower() for e in elements]:
        elements.append(user_level)  # Ensure the user level is included

    return elements

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=["POST"])
def submit():
    data = request.json
    user_input = data.get("input", "").strip()

    if not user_input:
        return jsonify({"error": "Please enter a valid response."}), 400

    if user_session["query"] is None:
        user_session.update({"query": user_input, "responses": [], "questions": [], "question_count": 1, "scores": []})
        first_question = generate_question()
        user_session["questions"].append(first_question)
        return jsonify({"question": first_question, "clear_input": True})

    last_question = user_session["questions"][-1]
    user_session["responses"].append(user_input)

    score, reasoning, inferred_level, more_questions_needed = score_response(user_input, last_question)
    user_session["scores"].append({"response": user_input, "score": score, "reasoning": reasoning})

    if len(user_session["responses"]) >= MIN_QUESTIONS and (more_questions_needed == "No" or len(user_session["responses"]) >= MAX_QUESTIONS):
        summary = generate_summary(user_session["responses"])
        key_elements = extract_key_elements(summary, inferred_level)

        final_response = {
            "result": f"User Level: {inferred_level}",
            "score_breakdown": user_session["scores"],
            "summary": summary,
            "key_elements": key_elements
        }

        print(f"DEBUG: Final JSON Response - {final_response}")  # 🔹 Check if everything is included

        return jsonify(final_response)
    
   

    next_question = generate_question()
    user_session["questions"].append(next_question)
    user_session["question_count"] += 1
    return jsonify({"question": next_question, "clear_input": True})


if __name__ == "__main__":
    app.run(debug=True)
