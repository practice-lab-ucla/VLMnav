import logging
import os
import torch
import numpy as np
import google.generativeai as genai
import cv2
import openai
import io
import base64
from dotenv import load_dotenv


from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline



load_dotenv()

CHATGPT_HISTORY_ACCESS = int(os.getenv("CHATGPT_HISTORY_ACCESS", 0))  # Default to 0 (No history)


api_key = os.environ.get("CHATGPT_API_KEY")
if not api_key:
    raise ValueError("❌ ERROR: CHATGPT_API_KEY is not set. Please check your .env file.")


class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent. 
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError
    
    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """

    def __init__(self, model="gemini-1.5-flash", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        # Configure generation parameters
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000
        
        # Initialize Gemini model and chat session
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=self.generation_config,
            system_instruction=system_instruction
        )
        self.session = self.model.start_chat(history=[])

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the Gemini model.

        Parameters
        ----------
        history : int
            The number of environment steps to keep in context.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.session.send_message([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

            # Manage history length based on the number of past steps to keep
            if history == 0:
                self.session = self.model.start_chat(history=[])
            elif len(self.session.history) > 2 * history:
                self.session.history = self.session.history[-2 * history:]
        
        except Exception as e:  
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text
    
    def rewind(self):
        """
        Rewind the chat history by one step.
        """
        if len(self.session.history) > 1:
            self.model.rewind()

    def reset(self):
        """
        Reset the chat history.
        """
        self.session = self.model.start_chat(history=[])

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform contextless inference with the Gemini model.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.
        """
        images = [Image.fromarray(image[:, :, :3], mode='RGB') for image in images]
        try:
            response = self.model.generate_content([text_prompt] + images)
            self.spend += (response.usage_metadata.prompt_token_count * self.cost_per_input_token +
                           response.usage_metadata.candidates_token_count * self.cost_per_output_token)

        except Exception as e:  
            logging.error(f"GEMINI API ERROR: {e}")
            return "GEMINI API ERROR"

        return response.text
    
    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend







class ChatGPTVLM(VLM):
    """
    A specific implementation of a VLM using OpenAI's ChatGPT API for text and image inference.
    """

    def __init__(self, model="gpt-4-vision-preview", system_instruction=None, max_history=5):
        """
        Initialize the ChatGPT model with specified configuration.

        Parameters
        ----------
        model : str
            The OpenAI model version to be used (must support vision for image input).
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.model = model
        self.system_instruction = system_instruction or "You are an AI assistant that processes text and images."
        self.spend = 0
        self.client = openai.OpenAI(api_key=api_key)  # Define OpenAI client inside class
        
        # store history if = 1
        self.history_enabled = CHATGPT_HISTORY_ACCESS == 1  # Enable history only if CHATGPT_HISTORY_ACCESS=1
        self.max_history = max_history    
        self.history = [] if self.history_enabled else None  # Store past interactions only if enabled
    
        if self.history_enabled:
            print("🟢 History is enabled for ChatGPT")


    
    def _encode_image(self, image: np.array):
        """
        Converts an image (numpy array) into a base64 string for OpenAI API.

        Parameters
        ----------
        image : np.array
            An RGB image.

        Returns
        -------
        str
            A base64-encoded string of the image.
        """
        pil_image = Image.fromarray(image[:, :, :3])  # Convert to PIL Image
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")  # Save as PNG
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference using OpenAI's ChatGPT model with optional history.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.

        Returns
        -------
        str
            The response from ChatGPT.
        """
        # Supress unnecessary print
        logging.getLogger("httpx").setLevel(logging.WARNING) 

        try:
            messages = [{"role": "system", "content": self.system_instruction}]

            # Include history if enabled
            if self.history_enabled and self.history:
                messages += self.history  

            user_message = {"role": "user", "content": text_prompt}
            if images:
                user_message["content"] = [{"type": "text", "text": text_prompt}]
                user_message["content"] += [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{self._encode_image(img)}"}}
                    for img in images
                ]

            messages.append(user_message)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            output = response.choices[0].message.content

            # Update history only if enabled
            if self.history_enabled:
                self.history.append(user_message)
                self.history.append({"role": "assistant", "content": output})

                # Limit history length
                if len(self.history) > 2 * self.max_history:
                    self.history = self.history[-2 * self.max_history:]

                # Print the current history length
                print(f"Current History Length: {len(self.history)}")

            return output

        except Exception as e:
            logging.error(f"❌ OpenAI API ERROR: {e}")
            return "OpenAI API ERROR"


    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with OpenAI's ChatGPT model.

        Parameters
        ----------
        history : int
            The number of past steps to keep in context (ignored for now).
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to process.

        Returns
        -------
        str
            The response from ChatGPT.
        """
        return self.call(images, text_prompt)

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend


















class DepthEstimator:
    """
    A class for depth estimation from images using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the depth estimation pipeline with the appropriate model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "Intel/zoedepth-kitti"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def call(self, image: np.array):
        """
        Perform depth estimation on an image.

        Parameters
        ----------
        image : np.array
            An RGB image for depth estimation.
        """
        original_shape = image.shape
        image_rgb = Image.fromarray(image[:, :, :3])
        depth_predictions = self.pipe(image_rgb)['predicted_depth']

        # Resize the depth map back to the original image dimensions
        depth_predictions = depth_predictions.squeeze().cpu().numpy()
        depth_predictions = cv2.resize(depth_predictions, (original_shape[1], original_shape[0]))

        return depth_predictions


class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask
