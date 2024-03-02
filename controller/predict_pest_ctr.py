from utils import read_file_as_image, preprocess_image, predict, generate_pest_control


async def predict_pest_ctr(file, location, language):
    # Load image
    image = await read_file_as_image(data=await file.read())    
    
    # Preprocess image
    preprocess_img = await preprocess_image(img=image)

    # Predict image
    predicted_class, confidence = await predict(preprocess_img)
    
    # Generate control solutions
    solutions = await generate_pest_control(pest=predicted_class, language=language, location=location)

    return {
        'pest': predicted_class,
        'confidence_score': confidence,
        'question': solutions.get('question'),
        'solutions': solutions.get('answer'),
        'sources':  solutions.get('sources')
    }