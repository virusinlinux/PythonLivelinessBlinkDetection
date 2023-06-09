import subprocess
import tkinter as tk
from PIL import Image, ImageTk
import cv2
import util
import os
import datetime
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import re
import uuid
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

class App:

    def __init__(self):
        #reqd for blink detection
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        self.main_window = tk.Tk()
        self.main_window.geometry('1200x520')
        # background image
        self.bg_frame = Image.open('myvenvpy/Resources/background.jpg')  # opens a image present in given path and stores it in variable self.bg_frame
        photo = ImageTk.PhotoImage(self.bg_frame)             # this line converts the image stored into bg_frame into a format which is compatible for tkinter to display and this converted photo is stroed in the variable named photo
        self.bg_panel = tk.Label(self.main_window,image=photo)         #this line creates a label named as bg_panel and sets its image attribute to photo
        self.bg_panel.image=photo                                   # This line stores the photo image in the image attribute of the self.bg_panel widget. This is done to prevent the image from being garbage collected by Python's memory management, as it only keeps a reference to the image object.
        self.bg_panel.pack(fill='both',expand='yes')                #Finally, the pack() method is used to add the self.bg_panel widget to the main window (self.main_window). The fill='both' parameter ensures that the widget fills the available space both horizontally and vertically, while expand='yes' allows the widget to expand if the window is resized.

        self.txt = "Welcome to FaceTrack: An Automated Check-In System!"
        self.heading = tk.Label(self.main_window, text=self.txt, font=('Arial',25,'bold'),fg='black',bg='yellow')
        self.heading.place(x=0,y=1,width=1240,height=30)
        # login & Register btn
        self.login_btn_main_window = util.get_button(self.main_window, 'Login', 'green', self.login)
        self.register_btn_main_window = util.get_button(self.main_window, 'Register', 'Blue', self.register)
        self.login_btn_main_window.place(x=800,y=100)
        self.register_btn_main_window.place(x=800,y=300)

        # Webcam Label & embed face using webcam in the label
        self.webcam_label = util.get_img_label(self.main_window)
        self.webcam_label.place(x=10,y=40,width=642,height=482)

        self.add_webCam(self.webcam_label)
        
        self.user_dir = './Users'
        if not os.path.exists(self.user_dir):
            os.makedirs(self.user_dir)

        # self.db_dir = './db'
        # if not os.path.exists(self.db_dir):
        #     os.mkdir(self.db_dir)


    def add_webCam(self,label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label

        self.process_webcam()

    # we have created this function to add the webcame frame into the label the frames that we read using opencv are in a different
    #format so we have to convert them into a format compatible by pillow we do this by using this function
    def process_webcam(self):
        
        ret, frame = self.cap.read()            # reading the frame from videocapture(0)
        self.most_recent_capture_arr = frame    # putting the frame in this instance var
        
        img =  cv2.cvtColor(self.most_recent_capture_arr,cv2.COLOR_BGR2RGB)   # converting the color from bgr to rgb
        self.most_recent_capture_pil = Image.fromarray((img))    #The converted RGB image is converted into a PIL (Python Imaging Library) Image object using Image.fromarray() method. This allows for further processing or displaying the image using PIL.
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        # The above line creates a Tkinter-compatible PhotoImage object imgtk from the PIL Image object self.most_recent_capture_pil. This prepares the image for display in a Tkinter GUI.
        self._label.imgtk = imgtk     # assigning the imagetk attribute of label to the new imgtk frame
        self._label.configure(image=imgtk) #assigning the img attribute of the label intsance the newly processed tk gui enabled image

        self._label.after(20,self.process_webcam)  # this function will be called again & again after 20 milliseconds to generate a live stream effect

    def eye_aspect_ratio(self,eye):
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
        blinked_frame=None

        # Perform blink detection for a certain duration
        while blinking_duration < 30:
            ret, frame = self.cap.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detector(gray,0)
            
            for rect in rects:
                shape = self.predictor(gray,rect)
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[36:42]
                right_eye = shape[42:48]
                
                left_ear = self.eye_aspect_ratio(left_eye)
                right_ear = self.eye_aspect_ratio(right_eye)
                
                avg_ear = (left_ear+right_ear)/2
                print("ear{}".format(avg_ear))
                
                if avg_ear<0.18:
                    blink_detected = True


            if blink_detected:
                    blinked_frame = frame                   
                    break

            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            blinked_time += 1
            print(blinked_time)
            blinking_duration = blinked_time //5   # Assuming 10 frames per second

        cv2.destroyAllWindows()

        return blink_detected,blinked_frame
                  
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
    
    def login(self):
#--------------------------------------blink check-------------------------------------------------
        #user-id verification
        self.login_window = tk.Toplevel(self.main_window)
        self.login_window.geometry('520x520')
        
        self.login_label = util.get_text_label(self.login_window,"enter user id:")
        self.login_label.place(x=100,y=100)
        
        self.login_text = util.get_entry_text(self.login_window)
        self.login_text.place(x=260,y=100)
        
        self.submit_btn = util.get_button(self.login_window,"Submit","Red",self.submit_action)
        self.submit_btn.place(x=120,y=300)
            
    def check_user(self,id):
        pass
    
    def try_again_action(self):
        self.register_window.destroy()

    def add_img_to_label(self,label):
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
        # The above line creates a Tkinter-compatible PhotoImage object imgtk from the PIL Image object self.most_recent_capture_pil. This prepares the image for display in a Tkinter GUI.
        label.imgtk = imgtk  # assigning the imagetk attribute of label to the new imgtk frame
        label.configure(image=imgtk)  # assigning the img attribute of the label intsance the newly processed tk gui enabled image

        self.register_new_user_capture = self.most_recent_capture_arr.copy()

        
        self.text_label_register_new_user = util.get_text_label(self.register_window,'Name:')
        self.text_label_register_new_user.place(x=800, y=30)
        self.enter_text_register = util.get_entry_text(self.register_window)
        self.enter_text_register.place(x=860,y=30)
        
        self.email_label = tk.Label(self.register_window,text="Email:")
        self.email_label.config(font=("sans-serif", 13), justify="left")
        self.email_label.place(x=800,y=80)
        self.email_entry_text = tk.Text(self.register_window,height=1,width=20, font=("Arial", 12))
        self.email_entry_text.place(x=860,y=80)

    def start(self):
        self.main_window.mainloop()

    def accept(self):
        name = self.enter_text_register.get(1.0,"end-1c")
        email = self.email_entry_text.get(1.0,"end-1c")
        isNameValid = self.validate_name(name)
        isEmailVAlid = self.validate_email(email)
        # print(name,email)
        
        if isNameValid and isEmailVAlid:
            #aage ka code 
            print("valid")
            self.generate_id(name,email)
        
    def generate_id(self,name,email):
        # Generate a unique ID for the user
        self.user_id = str(uuid.uuid4())
        # print(self.user_id)
        
        # Create a folder for the user
        folder_name = os.path.join(self.user_dir,f"User_{self.user_id}")
        os.makedirs(folder_name)
        
        #add the image into the folder
        cv2.imwrite(os.path.join(folder_name,'{}.jpg'.format(name)),self.register_new_user_capture)

        
        #create a file
        file_path = os.path.join(folder_name,"user_info.txt")
        with open(file_path,"w") as file:
            file.write(f'Name:{name}\nEmail:{email}')
            tk.messagebox.showinfo("Registration Successful",f"{name} has been registered successfully,  Kindly check your email for your new userid")
        file.close()
        self.register_window.destroy()
        self.user_email_registration(self.user_id,name,email)
        
    def user_email_registration(self,user_id,name,email):
        # print(f"email:{email}, name:{name},uid:{user_id}")
        sender_email = "rutupri123@gmail.com"
        receiver_email =  email
        subject = "Registration Successfull!"
        message = f"Dear {name} Congratulations! You have registered successfully, your unique user-id is:{user_id}.Thank you!!"

        
        # Create a multipart message object
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        
        # Attach the message to the email
        msg.attach(MIMEText(message, "plain"))
        
        # SMTP configuration
        smtp_host = "smtp.gmail.com"  # Replace with your SMTP host
        smtp_port = 587  # Replace with your SMTP port
        smtp_password = "vvgpsblurcmqfarh"  # Replace with your SMTP password
        
    
        try:
        # Create a SMTP session
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
            
                # Login to the SMTP server
                server.login("rutupri123@gmail.com", smtp_password)
            
                # Send the email
                server.sendmail(sender_email, receiver_email, msg.as_string())
        
            print("Registration email sent successfully!")
            
        except smtplib.SMTPException as e:
            print("Error sending registration email:", str(e))
            
    def send_email(self, recipient, subject, message):
            sender = 'rutupri123@gmail.com'  # Replace with your email address
            password = 'vvgpsblurcmqfarh'  # Replace with your email password

            # Compose the email
            email_text = f"Subject: {subject}\n\n{message}"

            # Send the email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender, password)
                server.sendmail(sender, recipient, email_text)

    def validate_name(self,name):
        if name=='':
            tk.messagebox.showerror("Invalid Name", "Please enter a valid name")
            self.register_window.destroy()
            return False
        
        return True
    
    def validate_email(self,email):
         # Check if email contains '@' symbol
        if '@' not in email:
            tk.messagebox.showerror("Invalid Email", "Email must contain '@' symbol.")
            self.register_window.destroy()
            return False
        return True

    def submit_action(self):
        self.id = self.login_text.get(1.0,'end-1c')
        # print(self.id)
        users_arr = os.listdir(self.user_dir)
        # print(users_arr)
        ids=[]
        for val in users_arr:
            ids.append(val[5:])
        
        # print(ids)
        if self.id in ids:
            print("user is registered")
            self.login_window.destroy()
            # Display a message indicating the user needs to blink
            util.msg_box("Blink", "Please blink to initiate login.")
            # Capture the frames and detect blink
            blink_detected,blinked_frame = self.detect_blink()
            
            self.photo_path = f'Users/User_{self.id}'
            #Users\User_078ce62e-80c7-4867-81ea-2332de1d520a
            print(self.photo_path)
            if blink_detected:
                util.msg_box("Success", "Blink detected,continuing to login.....")
                unknown_img_path = './.tmp.jpg'
                cv2.imwrite(unknown_img_path,blinked_frame)
                output = str(subprocess.check_output(['face_recognition', self.photo_path,unknown_img_path]))
                #now we have to parse the output to get the name
                
                name = output.split(',')[1][:-5]
                print(name)
                if name in ['unknown_person','no_persons_found']:
                    util.msg_box('Error','Unknown user, Please register or try again!')
                else:
                    util.msg_box('Success','Check-In successfull!! {} Your entry time has been recorded.'.format(name))
                    self.send_email('kumpz111@gmail.com', 'Attendance Confirmation',
                                f"Dear {name}, your attendance has been recorded.")
                    self.log_path=os.path.join(self.photo_path,'./Check-in-details.txt')
                    with open(self.log_path,'a') as f:
                        f.write('{},{}\n'.format(name,datetime.datetime.now()))
                        f.close()

                os.remove(unknown_img_path)

            else:
                util.msg_box("Error", "Blink not detected. Please try again.")
                cv2.destroyAllWindows()
        else:
            print("not registered")
            util.msg_box("Error","Invalid User Id,please try again.")
            self.login_window.destroy()
        
   


if __name__ == '__main__':
    app = App()
    app.start()
