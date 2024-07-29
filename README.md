# Flask Application with OpenAI Integration

This Flask application is designed to provide user authentication and integrates with OpenAI's GPT-4 for text processing. The application supports user login, signup, and document processing.

## Features

- **User Authentication**: Signup and login functionalities with password hashing.
- **Session Management**: User sessions are managed using Flask-Session.
- **OpenAI Integration**: Utilizes OpenAI's GPT-4 for processing documents.
- **Document Loading**: Supports loading and splitting PDF documents for text processing.

## Requirements

- Python 3.x
- Flask
- Flask-Session
- Werkzeug
- OpenAI's GPT-4 API Key
- Other dependencies listed in `requirements.txt`

## Installation

1. **Clone the repository**:

   ```sh
   git clone "https://github.com/Alvin-Alford/MigrAI.git"
   cd "./MigrAI"
   ```

2. **Create a virtual environment**:

   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```sh
   pip install -r requirements.txt
   ```

4. **Set environment variables**:
   Create a `.env` file in the root directory and add your OpenAI API key and Flask secret key:
   ```sh
   OPENAI_API_KEY=your_openai_api_key
   FLASK_SECRET_KEY=your_secret_key
   ```

## Usage

1. **Run the application**:

   ```sh
   python main.py
   ```

2. **Access the application**:
   Open your web browser and navigate to `http://127.0.0.1:5000`.

## File Structure

- `main.py`: Main application file containing the Flask routes and logic.
- `Userdata/users.json`: JSON file for storing user data.
- `Data/data.pdf`: PDF file to be processed by the application.
- `templates/`: Directory containing HTML templates for the application (login, signup, etc.).

## Routes

- **Home**: `GET /`
- **Login**: `GET, POST /login`
- **Signup**: `GET, POST /signup`
- **Logout**: `GET /logout`

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License.
