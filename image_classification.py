def clasify_accident(image_path)-> int:
    """Clasify the image passed
    Using a image recognition model clasify the severity of the car crash captured
    in the image. The accident severity is classified in 4 levels:
        0 - No Accident
        1 - Minor Accident
        2 - Moderate Accident
        3 - Severe Accident
    Args:
        image_path (string): Path to image

    Returns:
        int: Severity of the accident (0-3)
    """    