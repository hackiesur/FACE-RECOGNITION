import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import messagebox


if not os.path.exists('user_data'):
    os.mkdir('user_data')


def capture_face(username):
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the face image
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f'user_data/{username}.jpg', face_img)

            messagebox.showinfo("Face Capture", "Face captured successfully!")
            cap.release()
            cv2.destroyAllWindows()
            return

        cv2.imshow('Capture Face', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function for the sign-up process


def sign_up():
    username = username_entry.get()
    if not username:
        messagebox.showerror("Error", "Username cannot be empty.")
        return

    # Check if the user already exists
    if os.path.exists(f'user_data/{username}.jpg'):
        messagebox.showerror("Error", "Username already exists.")
        return

    capture_face(username)
    messagebox.showinfo("Success", "Sign-up successful!")

# Function for the sign-in process


def sign_in():
    username = username_entry.get()
    if not username:
        messagebox.showerror("Error", "Username cannot be empty.")
        return

    # Check if the user exists
    if not os.path.exists(f'user_data/{username}.jpg'):
        messagebox.showerror("Error", "Username not found.")
        return

    # Load the user's saved face image
    saved_face = cv2.imread(f'user_data/{username}.jpg', cv2.IMREAD_GRAYSCALE)

    # Capture a new face for comparison
    capture_face('temp')

    # Load the temporary captured face image
    temp_face = cv2.imread('user_data/temp.jpg', cv2.IMREAD_GRAYSCALE)

    # Compare the faces using a simple threshold
    similarity = np.mean(np.abs(saved_face - temp_face))

    if similarity < 20:
        messagebox.showinfo("Success", "Sign-in successful!")
    else:
        messagebox.showerror("Error", "Face recognition failed!")

# Create the main window


root = tk.Tk()
root.title("Face Sign-In/Sign-Up")

# Create username entry field
username_label = tk.Label(root, text="Username:")
username_label.pack()
username_entry = tk.Entry(root)
username_entry.pack()

# Create sign-up and sign-in buttons
sign_up_button = tk.Button(root, text="Sign Up", command=sign_up)
sign_up_button.pack()
sign_in_button = tk.Button(root, text="Sign In", command=sign_in)
sign_in_button.pack()

root.mainloop()
