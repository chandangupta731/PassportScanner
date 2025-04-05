from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
import easyocr
import os

app = Flask(__name__)

# Ensure the upload folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to format date
def format_date(date_str):
    try:
        year = int(date_str[:2])
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        year += 2000 if year < 50 else 1900
        return f"{day:02d}/{month:02d}/{year}"
    except:
        return ""


# Function to parse MRZ lines into structured data
def parse_mrz_to_dict(line1, line2):
    if len(line1) != 44 or len(line2) != 44:
        return {"error": "Invalid MRZ line lengths."}

    names = line1[5:].split("<<")
    last_name = names[0].replace('<', ' ').strip()
    first_name = names[1].replace('<', ' ').strip() if len(names) > 1 else ""

    return {
        "First Name": first_name,
        "Last Name": last_name,
        "Nationality": line2[10:13],
        "Date of Birth": format_date(line2[13:19]),
        "Date of Expiry": format_date(line2[21:27]),
    }


@app.route('/')
def index():
    # Serve the main HTML page
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    # Handling file upload
    file = request.files['file']
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    # Debug print to confirm the file save location
    print("Saved file to:", path)

    # Read the image with OpenCV
    image = cv2.imread(path)
    if image is None:
        return jsonify({"error": "Failed to load image with OpenCV"}), 400

    # Crop the MRZ area (bottom 30% of the image)
    height = image.shape[0]
    mrz_area = image[int(height * 0.70):, :]
    temp_crop = os.path.join(UPLOAD_FOLDER, "cropped.jpg")
    cv2.imwrite(temp_crop, mrz_area)

    # Use EasyOCR to read the MRZ text
    reader = easyocr.Reader(['en'], gpu=False)
    mrz_results = reader.readtext(temp_crop, detail=0)
    mrz_lines = [line.replace(" ", "").replace('\n', '') for line in mrz_results if line.count('<') > 5 and len(line) >= 30]

    # Check if MRZ lines are detected
    if len(mrz_lines) >= 2:
        data = parse_mrz_to_dict(mrz_lines[0], mrz_lines[1])
    else:
        data = {"error": "MRZ not detected"}

    # Return the parsed data as JSON
    return jsonify(data)


if __name__ == '__main__':
    # Dynamically assign the port for deployment (e.g., Render or Heroku)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)  # Keep debug=True if for development, else False for production
