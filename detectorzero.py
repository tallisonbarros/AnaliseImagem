
# ------------------- detectorzero.py
import FileFunctions
import ImageFunctions
import ProcessImageFunctions
import ui_detectorzero

print ("Iniciado \n")

pasta_amarelo = FileFunctions.Pasta("imagens", "calib_amarelo")
img_amarelo_0 = ImageFunctions.Imagem(pasta_amarelo.lista_arquivos[1])
img_amarelo_0.exibir()

pixel_amarelo = ProcessImageFunctions.contar_pixel_hsv(img_amarelo_0,(20, 100, 100),(35, 255, 255))
print ("\nPixels amarelos:", pixel_amarelo["percentual"],"%")

print ("\nFinalizado \n \n")