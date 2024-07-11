import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from config.config import EMAIL_CONFIG


class EmailAlertService(threading.Thread):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            # Email server connection
            server = smtplib.SMTP_SSL(
                EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]
            )
            server.login(EMAIL_CONFIG["from_email"], EMAIL_CONFIG["password"])

            # Compose email
            msg = MIMEMultipart()
            msg["From"] = EMAIL_CONFIG["from_email"]
            msg["To"] = EMAIL_CONFIG["to_email"]
            msg["Subject"] = "Motion Detected"

            body = "Motion has been detected. Please check the attached image."
            msg.attach(MIMEText(body, "plain"))

            # Attach image
            with open(self.image_path, "rb") as fp:
                img = MIMEImage(fp.read())
            img.add_header(
                "Content-Disposition", "attachment", filename=self.image_path
            )
            msg.attach(img)

            # Send email
            server.send_message(msg)
            server.quit()

            print(f"Email alert sent to {EMAIL_CONFIG['to_email']}")

        except Exception as e:
            print(f"Error sending email: {e}")
