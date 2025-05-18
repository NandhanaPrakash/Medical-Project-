from flask import Flask, request, jsonify
from .core import retrieve_with_gan, process_response, generate_final_response

app = Flask(__name__)

@app.route('/chat', methods=['POST', 'GET'])  # Allow both methods
def chat_endpoint():
    if request.method == 'GET':
        return """
        <form method="POST">
            <textarea name="message" rows="4" cols="50"></textarea><br>
            <input type="submit" value="Ask Medbot">
        </form>
        """
    
    # Handle POST
    try:
        if request.is_json:
            input_data = request.get_json()
            query = input_data.get("message", "")
        else:
            query = request.form.get("message", "")
        
        if not query:
            return jsonify({"error": "Message is required"}), 400

        print(f"Processing query: {query}")  # Debug
        
        # Step 1: Retrieve relevant info
        raw_retrieved = retrieve_with_gan(query)
        if not raw_retrieved:
            raw_retrieved = "No specific medical information found"
            
        # Step 2: Process the response
        processed = process_response(raw_retrieved)
        
        # Step 3: Generate final response
        response = generate_final_response(query, processed)
        
        if not response:
            response = "I couldn't generate a response. Please try rephrasing your question."
            
        print(f"Generated response: {response}")  # Debug
        
        if request.is_json:
            return jsonify({"insight": response})
        return response
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8000)