import logging

# Format all loggers in the application to show the filename and linenumber
logging.basicConfig(
    format="%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
)
