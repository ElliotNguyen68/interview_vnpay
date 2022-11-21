
import os
import configparser
import socket
import sys
import glob


def __get_config_general() -> configparser.ConfigParser:
    """Init ConfigParser

    Returns:
        configparser.ConfigParser: ConfigParser object
    """
    config = configparser.ConfigParser()
    config.read('/home/locnt7'+ "/configuration.ini")
    return config


def get_config(group: str, item: str) -> str:
    """Get configuration information

    Args:
        group (str): With group/system to get info ?
        item (str): Which specific item in group to get info ?

    Returns:
        str: Infomation of config
    """
    return __get_config_general().get(group, item)