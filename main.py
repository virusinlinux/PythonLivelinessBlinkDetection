import subprocess
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import util
import os
import datetime
import dlib
import re
from scipy.spatial import distance as dist
from imutils import face_utils


class App:

    def __init__(self):
        # reqd for blink detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.main_window = tk.Tk()
        self.main_window.geometry('1200x520')
        # background image
        self.bg_frame = Image.open(
            'background.jpg')  # opens a image present in given path and stores it in variable self.bg_frame
        photo = ImageTk.PhotoImage(
            self.bg_frame)  # this line converts the image stored into bg_frame into a format which is compatible for tkinter to display and this converted photo is stroed in the variable named photo
        self.bg_panel = tk.Label(self.main_window,
                                 image=photo)  # this line creates a label named as bg_panel and sets its image attribute to photo
        self.bg_panel.image = photo  # This line stores the photo image in the image attribute of the self.bg_panel widget. This is done to prevent the image from being garbage collected by Python's memory management, as it only keeps a reference to the image object.
        self.bg_panel.pack(fill='both',
                           expand='yes')  # Finally, the pack() method is used to add the self.bg_panel widget to the main window (self.main_window). The fill='both' parameter ensures that the widget fills the available space both horizontally and vertically, while expand='yes' allows the widget to expand if the window is resized.

        self.txt = "Welcome to FaceTrack: An Automated Check-In System!"
        self.heading = tk.Label(self.main_window, text=self.txt, font=('Arial', 25, 'bold'), fg='black', bg='yellow')
        self.heading.place(x=0, y=1, width=1240, height=30)
        # login & Register btn
        self.login_btn_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.register_btn_main_window = util.get_button(self.main_window, 'Register', 'Blue', self.register)
        self.login_btn_main_window.place(x=800, y=100)
        self.register_btn_main_window.place(x=800, y=300)

        # Webcam Label & embed face using webcam in the label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10, y=40, width=642, height=482)

        self.add_webCam(self.webcam_label)

        self.db_dir = './db'
        if not os.path.exists(self.db_dir):
            os.mkdir(self.db_dir)

        self.log_path = './Check-in-details.txt'

    def add_webCam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label

        self.process_webcam()

    # we have created this function to add the webcame frame into the label the frames that we read using opencv are in a different
    # format so we have to convert them into a format compatible by pillow we do this by using this function
    def process_webcam(self):

        ret, frame = self.cap.read()  # reading the frame from videocapture(0)
        self.most_recent_capture_arr = frame  # putting the frame in this instance var

        img = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)  # converting the color from bgr to rgb
        self.most_recent_capture_pil = Image.fromarray((
                                                           img))  # The converted RGB image is converted into a PIL (Python Imaging Library) Image object using Image.fromarray() method. This allows for further processing or displaying the image using PIL.
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        # The above line creates a Tkinter-compatible PhotoImage object imgtk from the PIL Image object self.most_recent_capture_pil. This prepares the image for display in a Tkinter GUI.
        self._label.imgtk = imgtk  # assigning the imagetk attribute of label to the new imgtk frame
        self._label.configure(
            image=imgtk)  # assigning the img attribute of the label intsance the newly processed tk gui enabled image

        self._label.after(20,
                          self.process_webcam)  # this function will be called again & again after 20 milliseconds to generate a live stream effect

    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        a = dist.euclidean(eye[1], eye[5])
        b = dist.euclidean(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal eye landmark
        c = dist.euclidean(eye[0], eye[3])

        # Calculate the eye aspect ratio
        ear = (a + b) / (2.0 * c)

        return ear

    def detect_blink(self):
        blink_detected = False
        blinked_time = 0
        blinking_duration = 0
        blinked_frame = None

        # Perform blink detection for a certain duration
        while blinking_duration < 30:
            ret, frame = self.cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray, 0)

            for rect in rects:
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[36:42]
                right_eye = shape[42:48]

                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)

                avg_ear = (left_ear + right_ear) / 2
                print("ear{}".format(avg_ear))

                if avg_ear < 0.21:
                    blink_detected = True

            if blink_detected:
                blinked_frame = frame
                break

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            blinked_time += 1
            print(blinked_time)
            blinking_duration = blinked_time // 6 # Assuming 10 frames per second

        cv2.destroyAllWindows()

        return blink_detected, blinked_frame

    def login(self):
        # --------------------------------------blink check-------------------------------------------------

        # Display a message indicating the user needs to blink
        util.msg_box("Blink", "Please blink to initiate login.")

        # Capture the frames and detect blink
        blink_detected, blinked_frame = self.detect_blink()

        if blink_detected:
            util.msg_box("Success", "Blink detected,continuing to login.....")
            unknown_img_path = './.tmp.jpg'
            cv2.imwrite(unknown_img_path, blinked_frame)
            output = str(subprocess.check_output(['face_recognition', self.db_dir, unknown_img_path]))
            # now we have to parse the output to get the name

            name = output.split(',')[1][:-5]
            print(name)
            if name in ['unknown_person', 'no_persons_found']:
                util.msg_box('Error', 'Unknown user, Please register or try again!')
            else:
                util.msg_box('Success', 'Check-In successfull!! {} Your entry time has been recorded.'.format(name))
                with open(self.log_path, 'a') as f:
                    f.write('{},{}\n'.format(name, datetime.datetime.now()))
                    f.close()

            os.remove(unknown_img_path)

        else:
            util.msg_box("Error", "Blink not detected. Please try again.")
            cv2.destroyAllWindows()

    def register(self):
        self.register_window = tk.Toplevel(self.main_window)
        self.register_window.geometry('1200x520+200+100')

        self.accept_new_user = util.get_button(self.register_window, 'Accept new User', 'green', self.accept)
        self.accept_new_user.place(x=800, y=200)

        self.try_again = util.get_button(self.register_window, 'Try Again', 'red', self.try_again_action)
        self.try_again.place(x=800, y=300)

        self.capture_label = util.get_img_label(self.register_window)
        self.capture_label.place(x=10, y=40, width=642, height=482)

        self.add_img_to_label(self.capture_label)

    def try_again_action(self):
        self.register_window.destroy()

    def add_img_to_label(self, label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        self.register_new_user_capture = self.most_recent_capture_arr.copy()
        self.enter_text_register = util.get_entry_text(self.register_window)
        self.enter_text_register.place(x=800, y=50)
        self.text_label_register_new_user = util.get_text_label(self.register_window, 'Please enter your Name:')
        self.text_label_register_new_user.place(x=800, y=30)

    def start(self):
        self.main_window.mainloop()

    def accept(self):
        name = self.enter_text_register.get(1.0, "end-1c")
        if name == '':
            util.msg_box('Error', "Please enter a valid name and try again!")
            self.register_window.destroy()
        elif not re.match(r'^[a-zA-Z]+$', name):
            util.msg_box('Error', "Please enter a valid name with alphabetic characters only!")
        else:
            cv2.imwrite(os.path.join(self.db_dir, '{}.jpg'.format(name)), self.register_new_user_capture)
            util.msg_box('Success', "User was registered successfully")
            self.register_window.destroy()

if __name__ == '__main__':
    app = App()
    app.start()
