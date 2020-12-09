import argparse

from logger import logger


def main(args):
    print(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Myunghwa AI Vision Inspection Practice Module (AIR Lab, Korea Univ)")
    args = parser.parse_args()

    logger.info(f"Selected parameters : {args}")

    main(args=args)
