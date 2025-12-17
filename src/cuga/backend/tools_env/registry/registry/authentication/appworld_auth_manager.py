import httpx
from cuga.backend.tools_env.registry.registry.authentication.base_auth_manager import BaseAuthManager
from loguru import logger
from cuga.config import settings


class AppWorldAuthManager(BaseAuthManager):
    def __init__(self, base_url="http://localhost:" + str(settings.server_ports.apis_url)):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self._account_passwords = self._load_account_passwords()
        self.profile = self._get_user_profile()

    def _get_user_profile(self):
        url = f"{self.base_url}/supervisor/profile"
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url)
                r.raise_for_status()
                return r.json()
        except httpx.RequestError as e:
            print(f"Error fetching user profile: {e}")
            return None

    def _load_account_passwords(self) -> dict[str, str]:
        url = f"{self.base_url}/supervisor/account_passwords"
        try:
            with httpx.Client(timeout=10.0) as client:
                r = client.get(url)
                r.raise_for_status()
                return {
                    item["account_name"]: item["password"]
                    for item in r.json()
                    if item.get("account_name") and item.get("password")
                }
        except httpx.RequestError as e:
            print(f"Error fetching app credentials: {e}")
            return {}

    def _get_credentials(self, app_name: str) -> str | None:
        return self._account_passwords.get(app_name)

    def _fetch_token(self, app_name: str, password: str) -> dict:
        logger.debug("Fetching token..")
        url = f"{self.base_url}/{app_name}/auth/token"
        user_name = self.profile["phone_number"] if app_name == "phone" else self.profile["email"]
        logger.debug(f"username: {user_name}")
        logger.debug(f"password: {password}")
        with httpx.Client(timeout=10.0) as client:
            r = client.post(url, data={"username": user_name, "password": password})
            r.raise_for_status()
            return r.json()
