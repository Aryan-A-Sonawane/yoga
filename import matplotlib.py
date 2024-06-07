import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a new figure
fig, ax = plt.subplots(figsize=(10, 8))

# Add a title
ax.set_title('Yoga Pose Detection and Feedback System Flowchart', fontsize=16, pad=20)

# Hide axes
ax.axis('off')

# Define the boxes and their positions
boxes = [
    {"label": "Start: The system begins", "xy": (0.5, 0.9)},
    {"label": "Pose Detection: Utilize MediaPipe to detect key body landmarks in real-time", "xy": (0.5, 0.75)},
    {"label": "Machine Learning Model: Train a machine learning model to classify yoga poses based on the detected landmarks", "xy": (0.5, 0.6)},
    {"label": "Web-Based Deployment: Develop a web application to host the model", "xy": (0.5, 0.45)},
    {"label": "User Interaction: Users access the web app, perform yoga poses, and receive feedback", "xy": (0.5, 0.3)},
    {"label": "Real-Time Tracking: Continuously monitor user movements", "xy": (0.5, 0.15)},
    {"label": "Feedback: Provide instant feedback on alignment, balance, and posture", "xy": (0.5, 0.05)},
    {"label": "End: The system concludes", "xy": (0.5, -0.1)},
]

# Draw the boxes
for box in boxes:
    rect = patches.FancyBboxPatch(box['xy'], width=0.9, height=0.1, boxstyle="round,pad=0.1", edgecolor='black', facecolor='lightblue')
    ax.add_patch(rect)
    ax.text(box['xy'][0], box['xy'][1] + 0.05, box['label'], va='center', ha='center', fontsize=10)

# Draw the arrows
for i in range(len(boxes) - 1):
    start = (boxes[i]['xy'][0] + 0.45, boxes[i]['xy'][1])
    end = (boxes[i+1]['xy'][0] + 0.45, boxes[i+1]['xy'][1] + 0.1)
    ax.annotate('', xy=end, xytext=start, arrowprops=dict(arrowstyle='->', lw=1.5))

# Adjust layout
plt.tight_layout()

# Save the flowchart
flowchart_path = "/mnt/data/yoga_pose_detection_flowchart.png"
plt.savefig(flowchart_path)

# Display the flowchart
plt.show()

# Provide the path to the saved flowchart

