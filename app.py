from flask import Flask, render_template, request, jsonify, redirect, url_for, Response
import face_recognition
import cv2
import numpy as np
import os
import sqlite3
from datetime import datetime
import io
import csv

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'



def load_images():
    known_face_encodings = []
    known_face_names = []
    image_paths = []

    for image_name in os.listdir(app.config['UPLOAD_FOLDER']):
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_name)
        image = face_recognition.load_image_file(image_path)
        face_encoding = face_recognition.face_encodings(image)
        if face_encoding:
            known_face_encodings.append(face_encoding[0])
            known_face_names.append(image_name)
            image_paths.append(image_path)  # Save path to image

    return known_face_encodings, known_face_names, image_paths

known_face_encodings, known_face_names, image_paths = load_images()

def log_to_database(name):
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    
    # Get the current date and time
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    
    # Print values for debugging
    print(f"Logging to database: name={name}, date={date_str}, time={time_str}")
    
    # Insert the log into the database
    c.execute("INSERT INTO face_recognition_logs (name, date, time) VALUES (?, ?, ?)",
              (name, date_str, time_str))
    
    # Commit the changes and close the connection
    conn.commit()
    conn.close()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    image_file = request.files.get('image')
    if image_file:
        # Convert image file to numpy array
        img_arr = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

        # Find faces in the uploaded image
        rgb_img = img[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        matches = []
        for face_encoding in face_encodings:
            face_matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            if True in face_matches:
                first_match_index = face_matches.index(True)
                match_name = known_face_names[first_match_index]
                matches.append({
                    "name": match_name,
                    "image_path": image_paths[first_match_index]
                })
                log_to_database(match_name)  # Log the match to the database

        response = {"matches": matches}
        return jsonify(response)

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.lower().endswith(('jpg', 'jpeg')):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        global known_face_encodings, known_face_names, image_paths
        known_face_encodings, known_face_names, image_paths = load_images()  # Reload the images
        return redirect(url_for('index'))

@app.route('/view_logs')
def view_logs():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("SELECT * FROM face_recognition_logs")
    logs = c.fetchall()
    conn.close()
    return render_template('view_logs.html', logs=logs)

@app.route('/clear_logs', methods=['POST'])
def clear_logs():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("DELETE FROM face_recognition_logs")
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/export_logs')
def export_logs():
    conn = sqlite3.connect('face_recognition.db')
    c = conn.cursor()
    c.execute("SELECT * FROM face_recognition_logs")
    logs = c.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Date', 'Time'])

    for log in logs:
        writer.writerow(log)

    output.seek(0)

    return Response(output, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=face_recognition_logs.csv"})

if __name__ == '__main__':
    app.run(debug=True)
