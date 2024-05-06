"""
File description: Helper functions

Author: Michal Cejpek (xcejpe05@stud.fit.vutbr.cz)
"""


def verbose_print(message, verbose=True):
    """
    Print the message if verbose is True
    :param message: str: Message to print
    :param verbose: bool: Print the message if True
    """
    if verbose:
        print(message)
