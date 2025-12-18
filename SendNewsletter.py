import sys
import os
import datetime
import mailchimp_marketing as MailchimpMarketing
from mailchimp_marketing.api_client import ApiClientError

# ------------ CONFIG ------------
MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")  # Set your Mailchimp API key in environment variable
MAILCHIMP_SERVER = "us21"
LIST_ID = "a38116c1cf"
SEG_ID = 3015050
# --------------------------------

def send_html_file(path_to_html: str, subject_line: str):
    if not os.path.isfile(path_to_html):
        print(f"[send] file not found: {path_to_html}")
        return

    with open(path_to_html, "r", encoding="utf-8") as f:
        html = f.read()

    client = MailchimpMarketing.Client()
    client.set_config({"api_key": MAILCHIMP_API_KEY, "server": MAILCHIMP_SERVER})

    try:
        camp = client.campaigns.create({
            "type": "regular",
            "recipients": {"list_id": LIST_ID, "segment_opts": {"saved_segment_id": SEG_ID}},
            "settings": {"subject_line": subject_line, "from_name": "Differ Insights", "reply_to": "insights@differ.se"}
        })
        cid = camp["id"]
        client.campaigns.set_content(cid, {"html": html})
        client.campaigns.send(cid)
        print(f"[send] sent campaign {cid}")
    except ApiClientError as e:
        print(f"[send] mailchimp error: {e.text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
         print("Usage: python SendNewsletter.py /absolute/path/to/file.html [subject line]")
         sys.exit(1)
    # path = "/Users/axelww/Code/Monthly digest/Output/monthly_digest_20251024_140244.html"
    path = sys.argv[1]
    if len(sys.argv) >= 3:
        subject = sys.argv[2]
    else:
        subject = f"Differ — Monthly Brief ({datetime.date.today().strftime('%B %Y')})"

    send_html_file(path, subject)
