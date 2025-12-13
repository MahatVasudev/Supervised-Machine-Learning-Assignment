import sys
from datetime import datetime


class Logger:
    RESET = "\033[0m"
    RED = "\033[1;31m"
    GREEN = "\033[1;32m"
    BLUE = "\033[1;34m"
    YELLOW = "\033[1;33m"
    CYAN = "\033[1;36m"

    BLOCK = "\n\t"
    WARNING_ = f"{YELLOW}[warning]{RESET}:"
    INFO_ = f"{GREEN}[info]{RESET}:"
    ERROR = f"{RED}[error]{RESET}:"
    FATAL = f"{RED}[fatal error]{RESET}:"
    EXECUTED = f"{CYAN}[executed]{RESET}:"

    @staticmethod
    def _ts():
        return datetime.now().strftime("[%H:%M:%S] ")

    @classmethod
    def _custom(cls, status, *msg):
        separator = "\n\t"
        text = separator.join(str(m) for m in msg)
        return f"{cls._ts()}{status}{cls.BLOCK}{text}{cls.RESET}"

    @classmethod
    def warning(cls, *msg):
        print(cls._custom(cls.WARNING_, *msg))

    @classmethod
    def info(cls, *msg):
        print(cls._custom(cls.INFO_, *msg))

    @classmethod
    def error(cls, *msg):
        print(cls._custom(cls.ERROR, *msg))

    @classmethod
    def fatal(cls, *msg):
        print(cls._custom(cls.FATAL, *msg))
        sys.exit(1)

    @classmethod
    def executed(cls, *msg):
        print(cls._custom(cls.EXECUTED, *msg))

    @classmethod
    def info_line(cls, msg):
        print(f"{cls._ts()}{cls.INFO_} {msg}{cls.RESET}")

    @classmethod
    def warning_line(cls, msg):
        print(f"{cls._ts()}{cls.WARNING_} {msg}{cls.RESET}")
