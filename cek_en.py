from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import base64

def pad(data):
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    return padder.update(data) + padder.finalize()

def shorten_name(name, key):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = pad(name.encode('utf-8'))
    print(padded_data)
    encrypted_name = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(encrypted_name)[:8].decode('utf-8')  # Adjust the length as needed

# Example usage
parent_name = "John Doe"
encryption_key = b'0123456789abcdef'  # Use a valid key size: 16 bytes (128 bits)
shortened_name = shorten_name(parent_name, encryption_key)
print(shortened_name)