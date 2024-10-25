import requests

# Replace with your actual WordPress site and credentials
wordpress_url = "https://your-wordpress-site.com/wp-json/wp/v2/posts"
username = "your-username"
password = "your-password"  # You can generate an app-specific password

# Function to post the content
def post_to_wordpress(title, content):
    data = {
        "title": title,
        "content": content,
        "status": "publish"
    }

    response = requests.post(
        wordpress_url,
        json=data,
        auth=(username, password)
    )

    if response.status_code == 201:
        print("Post published successfully.")
    else:
        print(f"Failed to publish post: {response.status_code} {response.text}")

# Example usage
title = "AI-Generated Post Example"
content = "This is a test post generated by AI."

post_to_wordpress(title, content)