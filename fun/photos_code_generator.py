import os

# Get all .jpg and .png files in current directory
image_files = [f for f in os.listdir('photos/') if f.lower().endswith(('.jpg', '.png'))]

# Sort by filename in reverse order (most recent first)
image_files.sort(reverse=True)

# Print HTML for each image
for filename in image_files:
    print(f'<img src="photos/{filename}" class="centerImg" style="width: 75%">')
    print('<br>')
