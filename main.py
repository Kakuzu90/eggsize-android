import cv2
import numpy as np
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock  # Import Clock for scheduling
from kivy.graphics import Color, Rectangle
from kivy.core.window import Window
from kivy_garden.camera import Camera
# Set the background color to white
Window.clearcolor = (1, 1, 1, 1)

# Constants for ArUco marker
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
MARKER_SIZE_CM = 5.0  # Known size of the marker in cm

class CameraApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.image_widget = Image()
        
        # Create a styled button
        self.button = Button(
            text="CAPTURE & PROCESS",
            size_hint=(1, 0.1),
            background_color=(0, 0, 1, 1),  # Blue background
            background_normal='',  # Removes the default background image
            color=(1, 1, 1, 1),  # White text color
            font_name='Arial',  # Arial font
            font_size=20  # Font size
        )
        self.button.bind(on_press=self.capture_and_process)
        
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.image_widget)
        layout.add_widget(self.button)

        # Schedule the first frame update
        Clock.schedule_interval(self.update_camera_frame, 1 / 30.0)
        return layout

    def update_camera_frame(self, *args):
        ret, frame = self.capture.read()
        if ret:
            # Convert frame to Kivy texture
            buffer = cv2.flip(frame, 0).tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buffer, colorfmt='bgr', bufferfmt='ubyte')
            self.image_widget.texture = texture

    def capture_and_process(self, *args):
        ret, frame = self.capture.read()
        if ret:
            # Save the captured image
            filename = 'captured_image.jpg'
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")

            # Process the image to detect and measure the ArUco marker
            self.process_image(frame)

    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT, parameters=ARUCO_PARAMS)

        if ids is not None:
            # Draw detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            for corner, marker_id in zip(corners, ids):
                corners_reshaped = corner.reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = corners_reshaped

                # Calculate distances
                width = np.linalg.norm(top_right - top_left)
                height = np.linalg.norm(top_left - bottom_left)

                # Calculate size in cm
                pixel_cm_ratio = MARKER_SIZE_CM / max(width, height)
                object_width_cm = width * pixel_cm_ratio
                object_height_cm = height * pixel_cm_ratio

                print(f"Marker ID: {marker_id[0]}, Width: {object_width_cm:.2f} cm, Height: {object_height_cm:.2f} cm")

        else:
            print("No ArUco markers detected.")

        # Show processed image
        #cv2.imshow("Processed Image", frame)
        cv2.waitKey(3000)
        cv2.destroyAllWindows()

    def on_stop(self):
        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CameraApp().run()
