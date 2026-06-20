import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'D:\SOFTWARE\tesseract\tesseract.exe'
img = cv2.imread('escalated1_op.png')
text = pytesseract.image_to_string(img, lang='spa')
print(text)
print(text)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#si el color de pluma cambia el umbralizado debe de cambiar de parametros