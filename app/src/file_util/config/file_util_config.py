import os
from dotenv import load_dotenv

class FileUtilConfig:

    def __init__(self):
        load_dotenv()

        # ENABLE_SMB_CIFS
        self.enable_smb_cifs = os.getenv("ENABLE_SMB_CIFS", "false").lower() == "true"

        # SMB_CIFS_SERVER
        self.smb_cifs_server = os.getenv("SMB_CIFS_SERVER", "smb.example.com")

        # SMB_CIFS_SHARE
        self.smb_cifs_share = os.getenv("SMB_CIFS_SHARE", "shared")

        # SMB_CIFS_USERNAME
        self.smb_cifs_username = os.getenv("SMB_CIFS_USERNAME", "user")

        # SMB_CIFS_PASSWORD
        self.smb_cifs_password = os.getenv("SMB_CIFS_PASSWORD", "password")

