
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError

# ---- CONFIG ----
MAILCHIMP_API_KEY = "914174703889684b6f268d80d12f2d91-us21"
MAILCHIMP_SERVER = "us21"   # e.g. "us21"
TEMPLATE_ID = 10574664        # replace with your template ID
# ----------------


# Setup client
mc_client = MailchimpMarketing.Client()
mc_client.set_config({
    "api_key": MAILCHIMP_API_KEY,
    "server": MAILCHIMP_SERVER
})

template_content = mc_client.templates.get_default_content_for_template(TEMPLATE_ID)

if template_content.get("html"):
    print("Code-based template (full HTML editable)")
elif template_content.get("sections"):
    print("Drag-and-drop template (sections editable)")
    print("Editable sections:", list(template_content["sections"].keys()))
else:
    print("Unknown template type / locked content")
