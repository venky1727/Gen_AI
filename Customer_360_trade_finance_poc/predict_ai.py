from google.cloud import aiplatform_v1
 
def predict_from_vertex(endpoint_id, project, location, instance_list):
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = aiplatform_v1.PredictionServiceClient(client_options=client_options)
 
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
 
    response = client.predict(endpoint=endpoint, instances=instance_list)
    return response