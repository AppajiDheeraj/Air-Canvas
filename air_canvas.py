import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Canvas settings - ADJUSTED DIMENSIONS
canvas_width, canvas_height = 1600, 850
toolbar_height = 120  # Increased toolbar height for better visibility

# Create canvas and toolbar
canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
toolbar = np.ones((toolbar_height, canvas_width, 3), dtype=np.uint8) * 240

# Color definitions (RGB format)
COLOR_BLACK = (0, 0, 0)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_PURPLE = (255, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_PINK = (203, 192, 255)
COLOR_TEAL = (128, 128, 0)

# Tool icons and positions
class Tool:
    def __init__(self, name, icon_color, x_pos, width=90):
        self.name = name
        self.icon_color = icon_color
        self.x_pos = x_pos
        self.width = width
        self.right_pos = x_pos + width

# Define tools with their positions
tools = [
    Tool("BRUSH", COLOR_BLACK, 20),
    Tool("LINE", COLOR_BLUE, 120),
    Tool("RECT", COLOR_GREEN, 220),
    Tool("CIRCLE", COLOR_RED, 320),
    Tool("ERASE", COLOR_WHITE, 420),
    Tool("CLEAR", COLOR_BLACK, 520),
]

# Define color palette
colors = [
    Tool("BLACK", COLOR_BLACK, 650, 60),
    Tool("BLUE", COLOR_BLUE, 720, 60),
    Tool("GREEN", COLOR_GREEN, 790, 60),
    Tool("RED", COLOR_RED, 860, 60),
    Tool("YELLOW", COLOR_YELLOW, 930, 60),
    Tool("PURPLE", COLOR_PURPLE, 1000, 60),
    Tool("ORANGE", COLOR_ORANGE, 1070, 60),
    Tool("PINK", COLOR_PINK, 1140, 60),
    Tool("TEAL", COLOR_TEAL, 1210, 60),
    Tool("WHITE", COLOR_WHITE, 1280, 60),
]

# Initialize variables
current_tool = "BRUSH"
current_color = COLOR_BLACK
brush_thickness = 5
start_x, start_y = None, None
is_drawing = False
is_selecting = False
selection_time = 0
points = []  # Store all the drawing points
undo_stack = []  # Store canvas states for undo
max_undo_states = 10  # Limit to save memory

# Function to check if coordinates are within toolbar area
def is_in_toolbar(y):
    return y < toolbar_height

# Function to get selected tool
def get_selected_tool(x, y):
    if not is_in_toolbar(y):
        return None
    
    # Check tools
    for tool in tools:
        if tool.x_pos <= x <= tool.right_pos:
            return tool.name
    
    # Check colors
    for color in colors:
        if color.x_pos <= x <= color.right_pos:
            return color.name
    
    # Check thickness controls
    if canvas_width - 250 <= x <= canvas_width - 190 and 60 <= y <= 100:
        return "THICKNESS_MINUS"
    elif canvas_width - 180 <= x <= canvas_width - 120 and 60 <= y <= 100:
        return "THICKNESS_PLUS"
    
    # Check undo button
    if canvas_width - 110 <= x <= canvas_width - 20 and 60 <= y <= 100:
        return "UNDO"
    
    return None

# Function to get color by name
def get_color_by_name(name):
    for color in colors:
        if color.name == name:
            return color.icon_color
    return COLOR_BLACK

# Function to update toolbar UI
def update_toolbar():
    toolbar = np.ones((toolbar_height, canvas_width, 3), dtype=np.uint8) * 240
    
    # Draw title "AIR CANVAS" centered at the top
    title_text = "AIR CANVAS"
    title_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    title_x = (canvas_width - title_size[0]) // 2
    cv2.putText(toolbar, title_text, (title_x, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50, 50, 50), 3, cv2.LINE_AA)
    
    # Draw horizontal separator line
    cv2.line(toolbar, (0, 50), (canvas_width, 50), (180, 180, 180), 2)
    
    # Draw tool buttons
    for tool in tools:
        is_selected = tool.name == current_tool
        button_color = (220, 220, 220) if not is_selected else (180, 180, 180)
        
        cv2.rectangle(toolbar, (tool.x_pos, 60), (tool.right_pos, 100), button_color, -1)
        cv2.rectangle(toolbar, (tool.x_pos, 60), (tool.right_pos, 100), (100, 100, 100), 2)
        
        # Center text
        text_size = cv2.getTextSize(tool.name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = tool.x_pos + (tool.width - text_size[0]) // 2
        
        # Draw text
        cv2.putText(toolbar, tool.name, (text_x, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Draw color palette
    for color in colors:
        is_selected = color.icon_color == current_color
        # Draw color circle
        cv2.circle(toolbar, (color.x_pos + 30, 80), 20, color.icon_color, -1)
        
        # Draw highlight for selected color
        if is_selected:
            cv2.circle(toolbar, (color.x_pos + 30, 80), 23, (0, 0, 0), 2)
    
    # Draw status text
    status_text = f"Current Tool: {current_tool} | Color: {next((c.name for c in colors if c.icon_color == current_color), 'Custom')}"
    cv2.putText(toolbar, status_text, (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    
    # Add thickness controls (moved to right side) with LARGER FONT SIZE
    cv2.putText(toolbar, f"Thickness: {brush_thickness}", (canvas_width - 250, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1, cv2.LINE_AA)
    
    # Thickness control buttons - ALIGNED BUTTONS
    cv2.rectangle(toolbar, (canvas_width - 250, 60), (canvas_width - 190, 100), (200, 200, 200), -1)
    cv2.putText(toolbar, "-", (canvas_width - 230, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.rectangle(toolbar, (canvas_width - 180, 60), (canvas_width - 120, 100), (200, 200, 200), -1)
    cv2.putText(toolbar, "+", (canvas_width - 160, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Add undo button - ALIGNED WITH OTHER BUTTONS
    cv2.rectangle(toolbar, (canvas_width - 110, 60), (canvas_width - 20, 100), (180, 180, 220), -1)
    cv2.putText(toolbar, "UNDO", (canvas_width - 95, 85), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    
    return toolbar

# Function to index_raised (check if finger is raised)
def is_index_raised(landmarks):
    index_tip = landmarks[8]  # Index finger tip
    index_pip = landmarks[6]  # Index finger PIP joint
    middle_tip = landmarks[12]  # Middle finger tip
    
    # Check if index finger is extended but middle finger is not
    return index_tip.y < index_pip.y and middle_tip.y > index_pip.y

# Function to check if thumb and index finger are close (pinch gesture)
def is_pinch_gesture(landmarks):
    thumb_tip = landmarks[4]  # Thumb tip
    index_tip = landmarks[8]  # Index finger tip
    
    # Calculate distance between thumb tip and index tip
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
    
    # If distance is small enough, it's a pinch
    return distance < 0.05  # Adjust threshold as needed

# Function to save current canvas state for undo
def save_canvas_state():
    global undo_stack
    if len(undo_stack) >= max_undo_states:
        undo_stack.pop(0)  # Remove oldest state
    undo_stack.append(canvas.copy())

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Flip frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(frame_rgb)
    
    # Prepare display image (combine toolbar and canvas)
    display = np.vstack([update_toolbar(), canvas])
    
    # Check for hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
            
            # Get index finger tip position
            landmarks = hand_landmarks.landmark
            index_tip = landmarks[8]
            x = int(index_tip.x * frame.shape[1])
            y = int(index_tip.y * frame.shape[0])
            
            # Scale coordinates to match display dimensions
            x_canvas = int(x * canvas_width / frame.shape[1])
            y_canvas = int(y * (canvas_height + toolbar_height) / frame.shape[0])
            
            # Ensure coordinates are within bounds
            x_canvas = min(max(0, x_canvas), canvas_width - 1)
            y_canvas = min(max(0, y_canvas), canvas_height + toolbar_height - 1)
            
            # Check if index finger is in toolbar area
            if y_canvas < toolbar_height:
                # Handle toolbar interactions
                selected = get_selected_tool(x_canvas, y_canvas)
                
                if selected:
                    if not is_selecting:
                        is_selecting = True
                        selection_time = time.time()
                    elif time.time() - selection_time > 0.5:  # Selection delay
                        # Handle tool selection
                        if selected in [t.name for t in tools]:
                            current_tool = selected
                            if selected == "CLEAR":
                                save_canvas_state()
                                canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
                                current_tool = "BRUSH"  # Reset to brush after clearing
                        # Handle color selection
                        elif selected in [c.name for c in colors]:
                            current_color = get_color_by_name(selected)
                        
                        # Handle thickness controls
                        elif selected == "THICKNESS_MINUS":
                            brush_thickness = max(1, brush_thickness - 1)
                        elif selected == "THICKNESS_PLUS":
                            brush_thickness = min(30, brush_thickness + 1)
                        
                        # Handle undo
                        elif selected == "UNDO":
                            if undo_stack:
                                canvas = undo_stack.pop()
                        
                        is_selecting = False
                else:
                    is_selecting = False
            else:
                is_selecting = False
                y_adjusted = y_canvas - toolbar_height  # Adjust y coordinate for canvas
                
                # Check if index finger is raised
                if is_index_raised(landmarks):
                    # Drawing mode
                    if current_tool == "BRUSH":
                        if not is_drawing:
                            is_drawing = True
                            start_x, start_y = x_canvas, y_adjusted
                        else:
                            cv2.line(canvas, (start_x, start_y), (x_canvas, y_adjusted), 
                                    current_color, brush_thickness)
                            start_x, start_y = x_canvas, y_adjusted
                    
                    elif current_tool == "ERASE":
                        cv2.circle(canvas, (x_canvas, y_adjusted), brush_thickness * 2, 
                                  COLOR_WHITE, -1)
                    
                    # Shape preview
                    elif current_tool in ["LINE", "RECT", "CIRCLE"]:
                        if not is_drawing:
                            save_canvas_state()  # Save state before drawing
                            is_drawing = True
                            start_x, start_y = x_canvas, y_adjusted
                        else:
                            # Create a copy for preview
                            preview = canvas.copy()
                            
                            if current_tool == "LINE":
                                cv2.line(preview, (start_x, start_y), (x_canvas, y_adjusted), 
                                       current_color, brush_thickness)
                            elif current_tool == "RECT":
                                cv2.rectangle(preview, (start_x, start_y), (x_canvas, y_adjusted), 
                                            current_color, brush_thickness)
                            elif current_tool == "CIRCLE":
                                radius = int(((start_x - x_canvas) ** 2 + (start_y - y_adjusted) ** 2) ** 0.5)
                                cv2.circle(preview, (start_x, start_y), radius, current_color, brush_thickness)
                            
                            # Use preview for display
                            display = np.vstack([update_toolbar(), preview])

                # Check for pinch gesture to complete shape or end drawing
                elif is_pinch_gesture(landmarks) and is_drawing:
                    if current_tool == "LINE":
                        cv2.line(canvas, (start_x, start_y), (x_canvas, y_adjusted), 
                               current_color, brush_thickness)
                    elif current_tool == "RECT":
                        cv2.rectangle(canvas, (start_x, start_y), (x_canvas, y_adjusted), 
                                    current_color, brush_thickness)
                    elif current_tool == "CIRCLE":
                        radius = int(((start_x - x_canvas) ** 2 + (start_y - y_adjusted) ** 2) ** 0.5)
                        cv2.circle(canvas, (start_x, start_y), radius, current_color, brush_thickness)
                    
                    is_drawing = False
                elif not is_pinch_gesture(landmarks):
                    is_drawing = False
            
            # Draw cursor circle at index finger tip position
            cursor_color = (0, 0, 0) if current_color == COLOR_WHITE else (255, 255, 255)
            cv2.circle(display, (x_canvas, y_canvas), brush_thickness, cursor_color, 2)
            cv2.circle(display, (x_canvas, y_canvas), 2, (0, 0, 255), -1)
    else:
        is_drawing = False
        is_selecting = False
    
    # Get display dimensions
    disp_h, disp_w = display.shape[:2]
    
    # Create new composite display with instructions
    # Create a bottom panel and video panel
    video_panel_height = 240
    info_panel_height = 240
    total_width = disp_w
    
    # Create hand tracking panel (resize video to fit better)
    tracking_frame = cv2.resize(frame, (320, 240))
    video_panel = np.ones((video_panel_height, 350, 3), dtype=np.uint8) * 220
    video_panel[0:240, 15:335] = tracking_frame
    cv2.putText(video_panel, "Hand Tracking", (15, 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(video_panel, (14, 0), (336, 240), (100, 100, 100), 1)
    
    # Create instructions panel
    instructions_panel = np.ones((info_panel_height, total_width - 350, 3), dtype=np.uint8) * 220
    instructions = [
        "Instructions:",
        "- Use index finger to draw",
        "- Pinch to complete shapes",
        "- Hover over toolbar to select tools",
        "- Press 'z' for undo, 'c' for clear",
        "- Use '+' and '-' keys to adjust brush thickness",
        "- Press ESC to exit"
    ]
    
    # Draw instructions with larger font
    for i, line in enumerate(instructions):
        cv2.putText(instructions_panel, line, (20, 30 + i*28), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Create bottom row panel
    bottom_panel = np.ones((video_panel_height, total_width, 3), dtype=np.uint8) * 230
    bottom_panel[0:video_panel_height, 0:350] = video_panel
    bottom_panel[0:info_panel_height, 350:total_width] = instructions_panel
    
    # Combine display with bottom panel
    full_display = np.vstack([display, bottom_panel])
    
    # Show the result in a properly sized window
    cv2.namedWindow("Air Canvas", cv2.WINDOW_NORMAL)
    cv2.imshow("Air Canvas", full_display)
    cv2.resizeWindow("Air Canvas", disp_w, disp_h + video_panel_height)
    
    # Check for key presses
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('+'):
        brush_thickness = min(30, brush_thickness + 1)
    elif key == ord('-'):
        brush_thickness = max(1, brush_thickness - 1)
    elif key == ord('z'):  # Undo with 'z' key
        if undo_stack:
            canvas = undo_stack.pop()
    elif key == ord('c'):  # Clear with 'c' key
        save_canvas_state()
        canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

# Release resources
cap.release()
cv2.destroyAllWindows()
