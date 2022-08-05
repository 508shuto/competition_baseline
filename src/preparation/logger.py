
import os
import sys
from logging import (DEBUG, INFO, FileHandler, Formatter, StreamHandler,
                     getLogger)


def get_logger(log_name=__name__, save_path="./log.txt", msg_format="%(levelname)-8s %(asctime)s [%(name)s] %(message)s"):
    """Setting Logger's Name and Text Path. Return Logger.

    Args:
        log_name (str, optional): Log Name. Defaults to __name__.
        save_path (str, optional): Save Path. Defaults to './log.txt'.
        msg_format (str, optional): Message Format. Defaults to "%(levelname)s %(asctime)s [%(name)s] %(message)s".

    Returns:
        Logger: Logger
    """    
    # ディレクトリがなければ作成する（既にあれば無視）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Setting logger
    logger = getLogger(log_name)
    logger.setLevel(DEBUG)

    # Setting FileHandler
    # NOTE: ファイル出力用
    handler = FileHandler(save_path)
    handler.setLevel(DEBUG)
    formatter = Formatter(msg_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Setting StreamHandler
    # NOTE: ターミナル出力用
    sh = StreamHandler(sys.stdout)
    sh.setLevel(INFO)  # INFO: 工程確認用
    logger.addHandler(sh)
    
    return logger
