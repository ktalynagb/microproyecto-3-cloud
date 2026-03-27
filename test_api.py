import requests

# La URL pública de tu backend en Azure
url = "http://pcb-backend-aci.centralus.azurecontainer.io:8080/api/v1/infer"

# Tu llave de seguridad
headers = {
    "X-API-Key": "pcb-api-key-super-secreto"  # Revisa en tu backend.py si el header se llama así o "Authorization"
}

# Cargamos la imagen (reemplaza con el nombre real de tu archivo local)
image_path = "C:\\Users\\Katalina Garcia\\Downloads\\test\\WIN_20221017_19_36_32_Pro_jpg.rf.7ce2ed23b595f9ed924f7dd0612e1f2b.jpg"

with open(image_path, "rb") as image_file:
    # FastAPI normalmente espera que el campo del formulario se llame "file" o "images"
    archivos_a_enviar = {"files": (image_path, image_file, "image/jpeg")}
    
    print("Enviando imagen al backend...")
    response = requests.post(url, headers=headers, files=archivos_a_enviar)

print(f"Status Code: {response.status_code}")
print("Respuesta del servidor:")
print(response.json())