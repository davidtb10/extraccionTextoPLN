from transformers import AutoModelForCausalLM, AutoTokenizer, GPTJForCausalLM
import torch
import os
from parallelformers import parallelize

class GPT_J:
    def __init__(self):
        self.load_GPT_J()

    def load_GPT_J(self):
        save_directory = "saved"

        if(not os.path.exists(save_directory + '/tokenizer.json')):
            self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", force_download=True)
            self.tokenizer.save_pretrained(save_directory)
            print("Tokenizer descargado")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(save_directory)
            print("Tokenizer cargado")

        if (not os.path.exists(save_directory + '/pytorch_model.bin')):
            self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True, force_download=True)
            self.model.save_pretrained(save_directory)
            print("Modelo descargado")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(save_directory)
            print("Modelo cargado")

        parallelize(self.model, num_gpus=2, fp16=True)
        print("Hecho el parallelize")


    def inference(self, question):
        context = """
La enfermedad por coronavirus de 2019, más conocida como COVID-19, covid-19 o covid, e incorrectamente llamada neumonía por coronavirus, coronavirus o corona, es una enfermedad infecciosa causada por el SARS-CoV-2.

Produce síntomas que incluyen fiebre, tos, disnea (dificultad respiratoria), mialgia (dolor muscular) y fatiga. En casos graves se caracteriza por producir neumonía, síndrome de dificultad respiratoria aguda, sepsis y choque circulatorio. Choque séptico es la forma más común en estos casos, pero los otros tipos también pueden ocurrir. Por ejemplo, choque obstructivo puede resultar de embolia pulmonar, una complicación de Covid-19. Según la OMS, la infección es mortal entre el 0,5 % y el 1 % de los casos. No existe tratamiento específico; las medidas terapéuticas principales consisten en aliviar los síntomas y mantener las funciones vitales.

La transmisión del SARS-CoV-2 se produce mediante pequeñas gotas —microgotas de Flügge— que se emiten al hablar, estornudar, toser o espirar, que al ser despedidas por un portador (que puede no tener síntomas de la enfermedad o estar incubándola) pasan directamente a otra persona mediante la inhalación, o quedan sobre los objetos y superficies que rodean al emisor, y luego, a través de las manos, que lo recogen del ambiente contaminado, toman contacto con las membranas mucosas orales, nasales y oculares, al tocarse la boca, la nariz o los ojos. También está documentada la transmisión por aerosoles ( < 5μm). La propagación mediante superficies contaminadas o fómites (cualquier objeto carente de vida, o sustancia, que si se contamina con algún patógeno es capaz de transferirlo de un individuo a otro) no contribuye sustancialmente a nuevas infecciones.

Los síntomas aparecen entre dos y catorce días (período de incubación), con un promedio de cinco días, después de la exposición al virus. Existe evidencia limitada que sugiere que el virus podría transmitirse uno o dos días antes de que se tengan síntomas, ya que la viremia alcanza un pico al final del período de incubación. El contagio se puede prevenir con el lavado de manos frecuente, o en su defecto la desinfección de las mismas con alcohol en gel, cubriendo la boca al toser o estornudar, ya sea con la sangradura (parte hundida del brazo opuesta al codo) o con un pañuelo y evitando el contacto cercano con otras personas, entre otras medidas profilácticas, como el uso de mascarillas. La OMS desaconsejaba en marzo la utilización de máscara quirúrgica por la población sana, en abril la OMS consideró que era una medida aceptable en algunos países. No obstante, ciertos expertos recomiendan el uso de máscaras quirúrgicas basados en estudios sobre la Influenza H1N1, donde muestran que podrían ayudar a reducir la exposición al virus. Los Centros para el Control y Prevención de Enfermedades (CDC) de Estados Unidos recomiendan el uso de mascarillas de tela, no médicas. Recomendación de los CDC (febrero de 2021).

El 12 de enero de 2020, la Organización Mundial de la Salud (OMS) recibió el genoma secuenciado del nuevo virus causante de la enfermedad y lo nombró temporalmente 2019-nCoV, del inglés 2019-novel coronavirus (nuevo coronavirus), mientras que la enfermedad era llamada «infección por 2019-nCoV» en documentos médicos, y SARS de Wuhan o Wu Flu (gripe de Wu) en Internet. El 30 de enero, la OMS recomendó que el nombre provisorio de la enfermedad fuera "enfermedad respiratoria aguda por 2019-nCoV", hasta que la Clasificación Internacional de Enfermedades diera un nombre oficial. A pesar de esta recomendación, los medios y agencias de noticias continuaron usando la denominación neumonía de Wuhan para referirse a la enfermedad.

La OMS anunció el 11 de febrero de 2020 que COVID-19 sería el nombre oficial de la enfermedad. El nombre es un acrónimo de coronavirus disease 2019 (enfermedad por coronavirus 2019, en español). Se procuró que la denominación no contuviera nombres de personas o referencias a ningún lugar, especie animal, tipo de comida, industria, cultura o grupo de personas, en línea con las recomendaciones internacionales, para evitar que hubiera estigmatización contra algún colectivo.

En diciembre de 2019 hubo un brote epidémico de neumonía de causa desconocida en Wuhan, provincia de Hubei, China; el cual, según afirmó más tarde Reporteros Sin Fronteras, llegó a afectar a más de 60 personas el día 20 de ese mes.
        """
        prompt = "Contexto:\n" + context + "\nPregunta:\n" + question + "\n\nRespuesta:\n"

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        gen_tokens = self.model.generate(input_ids, max_new_tokens=400)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        answer = gen_text.split("Respuesta:\n")[1]
        return answer
