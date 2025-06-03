# prompt: yesterday I was able to run using this command through local tunnel other script same way want to run this script
# !pip install streamlit localtunnel
#  !streamlit run mygan_app.py & npx localtunnel --port 8501

# Install necessary packages (Streamlit and localtunnel)
!pip install streamlit localtunnel

# Run Streamlit app in the background and expose it with localtunnel
# The '&' runs the streamlit command in the background.
# npx localtunnel --port 8501 --subdomain <your_optional_subdomain>
# Replace <your_optional_subdomain> with a unique name if you want a stable URL (optional)
# Add --subdomain if you want a stable URL, e.g., --subdomain my-digit-app
!streamlit run app.py & npx localtunnel --port 8501