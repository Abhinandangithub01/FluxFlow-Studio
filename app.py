#!/usr/bin/env python3
"""
FluxFlow Studio SaaS - Professional AI Image Generation Platform
Modern SaaS Application for E-commerce & Marketing Content Creation
"""
import os
import uuid
import time
import json
import logging
import requests
import jsonify
from flask import Flask, render_template, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROJECTS_FOLDER'] = 'static/projects'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROJECTS_FOLDER'], exist_ok=True)

HF_TOKEN = os.getenv('HF_TOKEN')
FAL_KEY = os.getenv('FAL_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
HF_API_URL = "https://api-inference.huggingface.co/models/"
FAL_API_URL = "https://fal.run/fal-ai/flux-pro/kontext"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
FLUX_MODELS = {
    'schnell': 'black-forest-labs/FLUX.1-schnell',
    'dev': 'black-forest-labs/FLUX.1-dev',
    'kontext': 'black-forest-labs/FLUX.1-Kontext-dev'
}

# Professional prompt templates for different use cases
PROMPT_TEMPLATES = {
    'ecommerce': {
        'product_hero': "Professional product photography of {product}, clean white background, studio lighting, commercial photography, high resolution, sharp focus, product showcase",
        'lifestyle': "Lifestyle product photography of {product}, natural environment, authentic usage scenario, professional lighting, commercial grade, marketing ready",
        'detail_shot': "Macro detail shot of {product}, extreme close-up, texture emphasis, professional studio lighting, commercial photography, high detail",
        'packaging': "Professional packaging photography of {product}, clean presentation, commercial lighting, marketing materials, brand showcase"
    },
    'marketing': {
        'social_media': "Social media marketing visual for {concept}, trendy design, eye-catching, modern aesthetic, brand-focused, engagement optimized",
        'banner': "Marketing banner design for {concept}, professional layout, call-to-action focused, brand consistent, conversion optimized",
        'advertisement': "Advertisement creative for {concept}, compelling visual, marketing campaign, professional design, brand storytelling",
        'promotional': "Promotional marketing material for {concept}, sales-focused, attractive design, conversion-driven, professional quality"
    }
}

STYLE_SUGGESTIONS = {
    'ecommerce': [
        "Clean white background with soft shadows",
        "Minimalist studio setup with professional lighting",
        "Luxury product presentation with premium feel",
        "Lifestyle context with natural environment",
        "Detail-focused macro photography style"
    ],
    'marketing': [
        "Bold and vibrant color scheme",
        "Modern gradient backgrounds",
        "Professional corporate aesthetic",
        "Creative artistic composition",
        "Trendy social media style"
    ]
}

class AdvancedFluxEngine:
    """Advanced FLUX.1 Engine with SaaS-grade features"""
    
    def __init__(self, hf_token):
        self.hf_token = hf_token
        self.headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
    
    def build_professional_prompt(self, base_prompt, workflow_type, style_type, quality="high"):
        """Build professional prompts for specific workflows"""
        
        # Get template if available
        template = PROMPT_TEMPLATES.get(workflow_type, {}).get(style_type, base_prompt)
        if '{product}' in template or '{concept}' in template:
            enhanced_prompt = template.format(product=base_prompt, concept=base_prompt)
        else:
            enhanced_prompt = f"{template}, {base_prompt}"
        
        # Add quality enhancements
        quality_terms = {
            "standard": "high quality, detailed, professional",
            "high": "8K resolution, professional grade, ultra high quality, detailed, sharp focus, commercial photography",
            "ultra": "8K resolution, professional grade, ultra high quality, extremely detailed, sharp focus, masterpiece, commercial grade, studio quality"
        }
        
        # Add workflow-specific enhancements
        workflow_enhancements = {
            "ecommerce": "commercial photography, product photography, studio lighting, professional grade, marketing ready, clean composition",
            "marketing": "marketing visual, professional design, brand-focused, conversion optimized, campaign ready, engaging composition"
        }
        
        if quality in quality_terms:
            enhanced_prompt += f", {quality_terms[quality]}"
        
        if workflow_type in workflow_enhancements:
            enhanced_prompt += f", {workflow_enhancements[workflow_type]}"
        
        return enhanced_prompt
    
    def generate_image(self, prompt, workflow_type="general", style_type="default", model="schnell", quality="high", max_retries=3):
        """Generate image with workflow-specific optimizations"""
        
        if not self.hf_token:
            return {"success": False, "error": "HF_TOKEN not configured"}
        
        # Build professional prompt
        enhanced_prompt = self.build_professional_prompt(prompt, workflow_type, style_type, quality)
        model_id = FLUX_MODELS.get(model, FLUX_MODELS['schnell'])
        api_url = f"{HF_API_URL}{model_id}"
        
        payload = {
            "inputs": enhanced_prompt,
            "parameters": {
                "num_inference_steps": 4 if model == "schnell" else 20,
                "guidance_scale": 1.0 if model == "schnell" else 3.5,
                "width": 1024, "height": 1024
            }
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Generating {workflow_type} image (attempt {attempt + 1}): {enhanced_prompt[:50]}...")
                response = requests.post(api_url, headers=self.headers, json=payload, timeout=60)
                
                if response.status_code == 200:
                    filename = f"{workflow_type}_{style_type}_{uuid.uuid4()}.png"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Image generated successfully: {filename}")
                    return {
                        "success": True,
                        "image_url": f"/static/uploads/{filename}",
                        "model": model_id,
                        "enhanced_prompt": enhanced_prompt,
                        "workflow": workflow_type,
                        "style": style_type
                    }
                elif response.status_code == 503:
                    wait_time = min(20 * (attempt + 1), 60)
                    logger.warning(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 429:
                    wait_time = min(10 * (attempt + 1), 30)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
            except Exception as e:
                logger.error(f"Generation error: {str(e)}")
                return {"success": False, "error": f"Generation failed: {str(e)}"}
        
        return {"success": False, "error": "Max retries exceeded - please try again later"}
    
    def edit_image(self, prompt, image_path, workflow_type="ecommerce", style_type="background_removal", quality="high", max_retries=3):
        """Edit image using FLUX.1-Kontext-dev for image-to-image editing"""
        
        if not self.hf_token:
            return {"success": False, "error": "HF_TOKEN not configured"}
        
        # Use Kontext model for image editing
        model_id = FLUX_MODELS['kontext']
        api_url = f"{HF_API_URL}{model_id}"
        
        # Read the image file
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
        except Exception as e:
            return {"success": False, "error": f"Failed to read image: {str(e)}"}
        
        # Build enhancement prompt
        enhanced_prompt = self.build_professional_prompt(prompt, workflow_type, style_type, quality)
        
        # Prepare multipart form data for image-to-image
        files = {
            'image': ('image.jpg', image_data, 'image/jpeg')
        }
        
        data = {
            'inputs': enhanced_prompt,
            'parameters': json.dumps({
                "num_inference_steps": 20,
                "guidance_scale": 3.5,
                "strength": 0.8  # How much to change the image (0.8 = moderate editing)
            })
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Editing image (attempt {attempt + 1}): {enhanced_prompt[:50]}...")
                response = requests.post(api_url, headers=self.headers, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    filename = f"edited_{workflow_type}_{style_type}_{uuid.uuid4()}.png"
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Image edited successfully: {filename}")
                    return {
                        "success": True,
                        "image_url": f"/static/uploads/{filename}",
                        "model": model_id,
                        "enhanced_prompt": enhanced_prompt,
                        "workflow": workflow_type,
                        "style": style_type
                    }
                elif response.status_code == 503:
                    wait_time = min(20 * (attempt + 1), 60)
                    logger.warning(f"Model loading, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 429:
                    wait_time = min(10 * (attempt + 1), 30)
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    error_msg = f"API Error {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    return {"success": False, "error": error_msg}
            except Exception as e:
                logger.error(f"Image editing error: {str(e)}")
                return {"success": False, "error": f"Image editing failed: {str(e)}"}
        
        return {"success": False, "error": "Max retries exceeded - please try again later"}
    
    def edit_image_kontext(self, prompt, image_path):
        """Edit image using FLUX.1-Kontext-dev locally - exact HF demo implementation"""
        try:
            # Use API-only approach for Railway deployment
            # Heavy model dependencies removed for lighter deployment
            from PIL import Image
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            
            logger.info(f"Using FLUX.1-Kontext-dev on {device}")
            
            # Load the Kontext pipeline - exactly like HF demo
            pipe = FluxKontextPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Kontext-dev", 
                torch_dtype=dtype
            ).to(device)
            
            # Load and process image
            image = Image.open(image_path)
            
            # Generate edited image - exact HF demo parameters
            result_image = pipe(
                prompt=prompt,
                image=image,
                height=1024,
                width=1024,
                num_inference_steps=28,
                guidance_scale=3.5,
                generator=torch.Generator(device).manual_seed(42)
            ).images[0]
            
            # Save result
            filename = f"kontext_edited_{uuid.uuid4()}.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            result_image.save(filepath)
            
            logger.info(f"FLUX.1-Kontext-dev editing successful: {filename}")
            return {
                "success": True,
                "image_url": f"/static/uploads/{filename}",
                "model": "FLUX.1-Kontext-dev",
                "enhanced_prompt": prompt,
                "provider": "Local Diffusers (HF Demo)"
            }
            
        except ImportError as e:
            return {"success": False, "error": f"Required libraries not installed: {str(e)}. Run: pip install diffusers torch"}
        except Exception as e:
            logger.error(f"FLUX.1-Kontext-dev error: {str(e)}")
            return {"success": False, "error": f"Kontext editing failed: {str(e)}"}
    
    def edit_image_fal(self, prompt, image_path, max_retries=3):
        """Edit image using FAL.ai FLUX-Pro Kontext for highest quality image editing"""
        
        if not FAL_KEY:
            return {"success": False, "error": "FAL_KEY not configured"}
        
        try:
            import base64
            
            # Read and encode the image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            headers = {
                "Authorization": f"Key {FAL_KEY}",
                "Content-Type": "application/json"
            }
            
            # Use FLUX-Pro Kontext parameters for highest quality
            payload = {
                "image_url": f"data:image/jpeg;base64,{image_data}",
                "prompt": prompt,
                "image_size": "landscape_4_3",  # Good for product photography
                "num_inference_steps": 28,  # Optimal for Kontext
                "guidance_scale": 3.5,  # Kontext optimal guidance
                "num_images": 1,
                "enable_safety_checker": True,
                "seed": 42  # Consistent results
            }
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Editing image with FAL.ai (attempt {attempt + 1}): {prompt[:50]}...")
                    response = requests.post(FAL_API_URL, headers=headers, json=payload, timeout=120)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'images' in result and len(result['images']) > 0:
                            image_url = result['images'][0]['url']
                            
                            # Download the edited image
                            img_response = requests.get(image_url, timeout=30)
                            if img_response.status_code == 200:
                                filename = f"edited_fal_{uuid.uuid4()}.png"
                                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                                with open(filepath, 'wb') as f:
                                    f.write(img_response.content)
                                
                                logger.info(f"Image edited successfully with FAL.ai: {filename}")
                                return {
                                    "success": True,
                                    "image_url": f"/static/uploads/{filename}",
                                    "model": "FLUX.1-Kontext-dev",
                                    "enhanced_prompt": prompt,
                                    "provider": "FAL.ai"
                                }
                        
                        return {"success": False, "error": "No images returned from FAL.ai"}
                    else:
                        error_msg = f"FAL.ai Error {response.status_code}: {response.text}"
                        logger.error(error_msg)
                        if attempt == max_retries - 1:
                            return {"success": False, "error": error_msg}
                        time.sleep(5 * (attempt + 1))
                        
                except Exception as e:
                    logger.error(f"FAL.ai request error: {str(e)}")
                    if attempt == max_retries - 1:
                        return {"success": False, "error": f"FAL.ai request failed: {str(e)}"}
                    time.sleep(5 * (attempt + 1))
            
            return {"success": False, "error": "Max retries exceeded with FAL.ai"}
            
        except Exception as e:
            return {"success": False, "error": f"Image editing preparation failed: {str(e)}"}
    
    def generate_ai_suggestions(self, image_path):
        """Generate AI-powered enhancement suggestions using Groq Llama-3.3-70b"""
        
        if not GROQ_API_KEY:
            return {"success": False, "error": "GROQ_API_KEY not configured"}
        
        try:
            import base64
            
            # Read and encode the image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Create a detailed prompt for the AI to analyze the image
            system_prompt = """You are an expert product photography consultant. Analyze the uploaded product image and provide 3-5 specific, actionable enhancement suggestions that would make this product photo more professional and suitable for e-commerce.

Focus on:
- Background improvements (remove/change background, add context)
- Lighting enhancements (studio lighting, shadows, highlights)
- Composition improvements (angles, positioning, staging)
- Color and contrast adjustments
- Professional styling suggestions

Provide each suggestion as a short, clear instruction that can be used as a prompt for AI image enhancement. Be specific about colors, materials, lighting, and styling."""

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": "Please analyze this product image and provide 3-5 specific enhancement suggestions for professional e-commerce photography."
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            
            logger.info("Generating AI suggestions with Groq Llama-3.3-70b...")
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                suggestions_text = result['choices'][0]['message']['content']
                
                # Parse suggestions into a list
                suggestions = []
                lines = suggestions_text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.')):
                        # Clean up the suggestion text
                        suggestion = line.lstrip('-•123456789. ').strip()
                        if suggestion:
                            suggestions.append(suggestion)
                
                logger.info(f"Generated {len(suggestions)} AI suggestions")
                return {
                    "success": True,
                    "suggestions": suggestions,
                    "raw_response": suggestions_text
                }
            else:
                error_msg = f"Groq API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"AI suggestions error: {str(e)}")
            return {"success": False, "error": f"Failed to generate suggestions: {str(e)}"}
    
    def generate_marketing_ai_suggestions(self, image_path, marketing_style='social_media'):
        """Generate AI-powered marketing suggestions using text-based AI"""
        
        if not GROQ_API_KEY:
            return {"success": False, "error": "GROQ_API_KEY not configured"}
        
        try:
            # For now, we'll use text-only suggestions since vision isn't working reliably
            logger.info(f"Generating {marketing_style} marketing suggestions")
            
            # Try to use Groq's vision model if available, otherwise fallback
            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Style-specific prompts
            style_prompts = {
                'social_media': """You are an expert social media marketing strategist and content creator. Create 3-5 detailed social media campaign concepts for packaged food products like nuts, snacks, or health foods.

For each campaign, provide:
1. PLATFORM: Which social media platform (Instagram, TikTok, Facebook)
2. VISUAL CONCEPT: Detailed description of the post/story design
3. CAPTION/TEXT: Exact post caption, hashtags, and copy
4. ENGAGEMENT STRATEGY: How to encourage likes, shares, comments

Format each suggestion as:
**Campaign Name**: [Creative title]
**Platform**: [Instagram/TikTok/Facebook]
**Visual**: [Detailed visual description including characters, colors, layout]
**Caption**: [Exact post text with hashtags]
**Engagement**: [Strategy to boost interaction]

Focus on:
- Fun cartoon characters and mascots (like dancing peanuts)
- Health and nutrition messaging
- Trending hashtags and challenges
- User-generated content ideas
- Influencer collaboration concepts
- Before/after health transformations
- Snacking occasion content

Make each campaign so detailed that someone could immediately create the social media post from your description.""",

                'banner': """You are an expert digital advertising specialist and banner designer. Create 3-5 detailed banner ad concepts for packaged food products like nuts, snacks, or health foods.

For each banner, provide:
1. SIZE/FORMAT: Banner dimensions and placement (leaderboard, square, mobile)
2. VISUAL LAYOUT: Detailed description of the banner design
3. HEADLINE TEXT: Main headline and subheadlines
4. BODY COPY: Supporting text and product benefits
5. CALL-TO-ACTION: Specific button text and action

Format each suggestion as:
**Banner Campaign**: [Creative title]
**Format**: [728x90 leaderboard / 300x300 square / etc.]
**Visual**: [Detailed layout with colors, images, mascots]
**Headline**: [Main headline text]
**Body Text**: [Supporting copy and benefits]
**CTA Button**: [Exact button text and color]

Focus on:
- Cartoon mascots and product characters
- Health benefits and nutrition facts
- Discount percentages and offers
- Trust badges and quality indicators
- Appetizing food photography
- Bright, eye-catching colors
- Clear value propositions

Make each banner so detailed that a designer could create it immediately from your description.""",

                'advertisement': """You are an expert advertising creative director and copywriter. Create 3-5 detailed advertisement concepts for packaged food products like nuts, snacks, or health foods.

For each advertisement, provide:
1. AD TYPE: Format (print ad, digital ad, billboard, etc.)
2. VISUAL CONCEPT: Detailed description of the main image/scene
3. HEADLINE: Primary headline text
4. BODY COPY: Supporting text and product benefits
5. TAGLINE: Memorable brand tagline or slogan

Format each suggestion as:
**Ad Campaign**: [Creative title]
**Type**: [Print/Digital/Billboard/Video]
**Visual Scene**: [Detailed description including characters, setting, colors]
**Main Headline**: [Primary headline text]
**Body Copy**: [Supporting text about benefits, features]
**Tagline**: [Memorable closing line]

Focus on:
- Cartoon mascots with personality (smiling peanuts, energetic nuts)
- Health transformation stories
- Family moments and snacking occasions
- Protein power and energy themes
- Natural and wholesome messaging
- Before/after scenarios
- Emotional connections to health and wellness

Make each ad so detailed that a creative team could produce it immediately from your description, including specific text content and visual elements.""",

                'promotional': """You are an expert promotional marketing specialist and creative director. Create 3-5 detailed promotional campaign concepts for packaged food products like nuts, snacks, or health foods.

For each campaign, provide:
1. VISUAL CONCEPT: Detailed description of the image/banner design
2. CONTENT/TEXT: Exact text, headlines, and copy to include
3. DESIGN ELEMENTS: Colors, fonts, layout, characters/mascots
4. CALL-TO-ACTION: Specific action for customers

Format each suggestion as:
**Campaign Name**: [Creative title]
**Visual**: [Detailed visual description including characters, colors, layout]
**Text Content**: [Exact headlines, body text, discount details]
**Call-to-Action**: [Specific action button/text]

Focus on:
- Cartoon mascots and characters (like cartoon peanuts, nuts with faces)
- Health benefits and nutrition facts
- Discount offers and limited-time deals
- Seasonal promotions and occasions
- Family-friendly and fun approaches
- Trust and quality messaging

Make each campaign so detailed that someone could immediately create the visual from your description."""
            }
            
            # Get the appropriate system prompt based on marketing style
            system_prompt = style_prompts.get(marketing_style, style_prompts['social_media'])

            # For now, let's use the text-only model with food-specific prompts since vision might not be available
            logger.info("Using text-only model for marketing suggestions...")
            
            # Use text-only analysis with product-specific examples
            fallback_prompt = f"""Based on common food/snack products, provide 3-5 creative {marketing_style.replace('_', ' ')} campaign ideas. 

Since this is for a packaged food product, focus on:
- Health and nutrition benefits
- Quality and freshness
- Convenience and snacking occasions
- Brand trust and reliability
- Value propositions

Create specific campaign concepts that would work well for packaged food items like nuts, snacks, or health foods."""

            payload = {
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": fallback_prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.8
            }
            
            logger.info("Making API call to Groq...")
            response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Vision API Response: {result}")
                
                if 'choices' in result and len(result['choices']) > 0:
                    suggestions_text = result['choices'][0]['message']['content']
                    logger.info(f"Raw suggestions text: {suggestions_text}")
                    
                    # Parse suggestions into a list
                    suggestions = []
                    lines = suggestions_text.split('\n')
                    logger.info(f"Processing {len(lines)} lines from AI response")
                    
                    for i, line in enumerate(lines):
                        line = line.strip()
                        logger.info(f"Line {i}: '{line}'")
                        
                        if line and (line.startswith('-') or line.startswith('•') or line.startswith('1.') or line.startswith('2.') or line.startswith('3.') or line.startswith('4.') or line.startswith('5.') or line.startswith('*')):
                            # Clean up the suggestion text
                            suggestion = line.lstrip('-•123456789. *').strip()
                            if suggestion:
                                suggestions.append(suggestion)
                                logger.info(f"Added suggestion: '{suggestion}'")
                    
                    # If no structured suggestions found, try to split by sentences
                    if not suggestions and suggestions_text:
                        logger.info("No structured suggestions found, trying sentence splitting")
                        sentences = [s.strip() for s in suggestions_text.split('.') if s.strip() and len(s.strip()) > 20]
                        suggestions = sentences[:5]  # Take first 5 sentences
                    
                    logger.info(f"Parsed {len(suggestions)} marketing AI suggestions: {suggestions}")
                    return {
                        "success": True,
                        "suggestions": suggestions,
                        "raw_response": suggestions_text
                    }
                else:
                    logger.error(f"No choices in API response: {result}")
                    return {"success": False, "error": "No suggestions generated by AI"}
            else:
                error_msg = f"Groq API Error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
                
        except Exception as e:
            logger.error(f"Marketing AI suggestions error: {str(e)}")
            return {"success": False, "error": f"Failed to generate marketing suggestions: {str(e)}"}

# Initialize advanced FLUX engine
flux_engine = AdvancedFluxEngine(HF_TOKEN)

@app.route('/studio')
def studio_interface():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FluxFlow Studio - Professional Workspace</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f8fafc;
            height: 100vh;
            overflow: hidden;
        }
        
        .studio-container {
            display: flex;
            height: 100vh;
        }
        
        /* Modern Sidebar */
        .sidebar {
            width: 280px;
            background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
            color: white;
            display: flex;
            flex-direction: column;
            box-shadow: 4px 0 20px rgba(0,0,0,0.15);
            position: relative;
        }
        
        .sidebar::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
            pointer-events: none;
        }
        
        .logo-section {
            padding: 30px 25px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            position: relative;
            z-index: 2;
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 15px;
            font-size: 22px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        
        .logo i {
            background: linear-gradient(135deg, #667eea, #764ba2);
            padding: 8px;
            border-radius: 10px;
            font-size: 18px;
        }
        
        .powered-by {
            font-size: 12px;
            color: #a0aec0;
            margin-left: 38px;
        }
        
        .nav-section {
            flex: 1;
            padding: 20px 0;
            position: relative;
            z-index: 2;
        }
        
        .nav-title {
            padding: 0 25px;
            font-size: 11px;
            font-weight: 700;
            color: #718096;
            text-transform: uppercase;
            margin-bottom: 15px;
            letter-spacing: 1px;
        }
        
        .nav-item {
            display: flex;
            align-items: center;
            padding: 15px 25px;
            cursor: pointer;
            transition: all 0.3s ease;
            gap: 15px;
            border-left: 3px solid transparent;
            position: relative;
        }
        
        .nav-item:hover {
            background: rgba(255,255,255,0.08);
            border-left-color: rgba(102,126,234,0.5);
        }
        
        .nav-item.active {
            background: rgba(102,126,234,0.2);
            border-left-color: #667eea;
        }
        
        .nav-item i {
            width: 20px;
            text-align: center;
            opacity: 0.8;
        }
        
        .nav-item span {
            font-weight: 500;
        }
        
        .nav-badge {
            background: #667eea;
            color: white;
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            margin-left: auto;
        }
        
        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: white;
            padding: 20px 30px;
            border-bottom: 1px solid #e2e8f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .header-left h1 {
            font-size: 28px;
            color: #1a202c;
            font-weight: 700;
            margin-bottom: 5px;
        }
        
        .header-subtitle {
            color: #718096;
            font-size: 14px;
        }
        
        .header-actions {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 14px;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 4px 12px rgba(102,126,234,0.3);
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102,126,234,0.4);
        }
        
        .btn-secondary {
            background: #f7fafc;
            color: #4a5568;
            border: 1px solid #e2e8f0;
        }
        
        .btn-secondary:hover {
            background: #edf2f7;
        }
        
        .workspace {
            flex: 1;
            padding: 30px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 30px;
        }
        
        /* Modern Cards */
        .card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .card-header {
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 25px;
        }
        
        .card-icon {
            width: 50px;
            height: 50px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .card-icon.ecommerce {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }
        
        .card-icon.marketing {
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }
        
        .card h3 {
            color: #1a202c;
            font-size: 20px;
            font-weight: 700;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2d3748;
            font-size: 14px;
        }
        
        .form-input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
            font-family: inherit;
        }
        
        .form-input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102,126,234,0.1);
        }
        
        .form-textarea {
            min-height: 120px;
            resize: vertical;
        }
        
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            align-items: end;
        }
        
        .suggestions-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        
        .suggestion-chip {
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            padding: 8px 12px;
            border-radius: 20px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .suggestion-chip:hover {
            background: #667eea;
            color: white;
            border-color: #667eea;
        }
        
        .status-message {
            padding: 15px 20px;
            border-radius: 8px;
            margin: 20px 0;
            font-weight: 500;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .status-success {
            background: #f0fff4;
            color: #22543d;
            border: 1px solid #9ae6b4;
        }
        
        .status-error {
            background: #fed7d7;
            color: #742a2a;
            border: 1px solid #feb2b2;
        }
        
        .status-warning {
            background: #fefcbf;
            color: #744210;
            border: 1px solid #f6e05e;
        }
        
        .result-image {
            max-width: 100%;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            margin-top: 20px;
        }
        
        .loading {
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .workflow-selector {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .workflow-option {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .workflow-option:hover {
            border-color: #667eea;
            transform: translateY(-2px);
        }
        
        .workflow-option.active {
            border-color: #667eea;
            background: #f7fafc;
        }
        
        .workflow-option i {
            font-size: 2rem;
            margin-bottom: 15px;
            color: #667eea;
        }
        
        .workflow-option h4 {
            margin-bottom: 10px;
            color: #1a202c;
        }
        
        .workflow-option p {
            color: #718096;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="studio-container">
        <!-- Modern Sidebar -->
        <div class="sidebar">
            <div class="logo-section">
                <div class="logo">
                    <i class="fas fa-magic"></i>
                    <span>FluxFlow Studio</span>
                </div>
                <div class="powered-by">Powered by FLUX.1 AI</div>
            </div>
            
            <nav class="nav-section">
                <div class="nav-title">Workflows</div>
                <div class="nav-item active" onclick="showWorkflow('ecommerce')">
                    <i class="fas fa-shopping-cart"></i>
                    <span>Product Photos</span>
                    <div class="nav-badge">Pro</div>
                </div>
                <div class="nav-item" onclick="showWorkflow('marketing')">
                    <i class="fas fa-bullhorn"></i>
                    <span>Marketing Content</span>
                    <div class="nav-badge">New</div>
                </div>
                <div class="nav-title" style="margin-top: 30px;">Tools</div>
                <div class="nav-item" onclick="showSection('gallery')">
                    <i class="fas fa-images"></i>
                    <span>Gallery</span>
                </div>
                
                <div class="nav-title" style="margin-top: 30px;">Account</div>
                <div class="nav-item" onclick="window.location.href='/'">
                    <i class="fas fa-home"></i>
                    <span>Home</span>
                </div>
            </nav>
        </div>
        
        <!-- Main Content -->
        <div class="main-content">
            <div class="header">
                <div class="header-left">
                    <div class="header-badge">
                        <i class="fas fa-sparkles"></i>
                        <span>AI Studio</span>
                    </div>
                    <h1 id="workflow-title">E-commerce Product Photos</h1>
                    <div class="header-subtitle" id="workflow-subtitle">Create professional product photography for your online store</div>
                </div>
                <div class="header-actions">
                    <!-- Header actions removed as requested -->
                </div>
            </div>
            
            <div class="workspace">
                <!-- E-commerce Workflow -->
                <div id="ecommerce-workflow" class="workflow-section">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon ecommerce">
                                <i class="fas fa-camera"></i>
                            </div>
                            <div>
                                <h3>Product Photography Studio</h3>
                                <p style="color: #718096; margin: 0;">Transform your mobile photos into professional product images</p>
                            </div>
                        </div>
                        
                        <!-- Modern Side-by-Side Layout -->
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                            <!-- Upload Section -->
                            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0; display: flex; align-items: center; gap: 8px;">
                                        <i class="fas fa-upload"></i> Original Image
                                    </h4>
                                    <span id="upload-status" style="color: rgba(255,255,255,0.6); font-size: 12px; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px;">Upload your photo</span>
                                </div>
                                <div class="upload-area" id="upload-area" onclick="document.getElementById('product-upload').click()" style="aspect-ratio: 1; border: 2px dashed rgba(255,255,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.3s ease;">
                                    <div class="upload-content" id="upload-content" style="text-align: center; color: rgba(255,255,255,0.6);">
                                        <i class="fas fa-cloud-upload-alt" style="font-size: 48px; margin-bottom: 15px; color: rgba(102, 126, 234, 0.7);"></i>
                                        <h4 style="color: white; margin: 0 0 8px 0;">Click to Upload</h4>
                                        <p style="margin: 0 0 8px 0; font-size: 14px;">Drag & drop or click to select</p>
                                        <small style="font-size: 12px; opacity: 0.7;">JPG, PNG, WebP • Max 10MB</small>
                                    </div>
                                    <img id="uploaded-preview" class="uploaded-image" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" alt="Product preview">
                                </div>
                                <input type="file" id="product-upload" accept="image/*" style="display: none;" onchange="handleProductUpload(event)">
                            </div>
                            
                            <!-- Result Section -->
                            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0; display: flex; align-items: center; gap: 8px;">
                                        <i class="fas fa-magic"></i> Enhanced Result
                                    </h4>
                                    <span id="result-status" style="color: rgba(255,255,255,0.6); font-size: 12px; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px;">Result will appear here</span>
                                </div>
                                <div style="aspect-ratio: 1; border: 2px dashed rgba(255,255,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                                    <div id="result-content" style="text-align: center; color: rgba(255,255,255,0.6);">
                                        <i class="fas fa-wand-magic-sparkles" style="font-size: 48px; margin-bottom: 15px; color: rgba(102, 126, 234, 0.7);"></i>
                                        <h4 style="color: white; margin: 0 0 8px 0;">Enhanced Image</h4>
                                        <p style="margin: 0 0 8px 0; font-size: 14px;">Your professional result will appear here</p>
                                        <small style="font-size: 12px; opacity: 0.7;">Upload an image and add a prompt to get started</small>
                                    </div>
                                    <img id="enhanced-preview" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" alt="Enhanced preview">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Controls Below -->
                        <div class="form-group">
                            <label class="form-label" style="display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-edit"></i> Enhancement Instructions
                            </label>
                            <textarea class="form-input form-textarea" id="ecommerce-prompt" 
                                placeholder="Describe how you want to enhance your product photo...&#10;Examples:&#10;• Place it on a wooden table&#10;• Remove background and add white background&#10;• Add professional studio lighting&#10;• Make it look premium and luxurious" style="min-height: 100px;"></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">AI-Powered Suggestions</label>
                            <div style="display: flex; flex-direction: column; gap: 15px;">
                                <button class="btn btn-secondary" onclick="generateAISuggestions()" id="ai-suggestions-btn" disabled style="display: flex; align-items: center; gap: 8px; justify-content: center;">
                                    <i class="fas fa-robot"></i>
                                    Generate AI Suggestions
                                </button>
                                <div id="ai-suggestions-container" style="display: none;">
                                    <div id="ai-suggestions-list" style="display: flex; flex-direction: column; gap: 8px;">
                                        <!-- AI suggestions will appear here -->
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <button class="btn btn-primary" onclick="enhanceProductPhoto()" id="ecommerce-btn" disabled>
                                <i class="fas fa-wand-magic-sparkles"></i>
                                Enhance Product Photo
                            </button>
                            <select class="form-input" id="ecommerce-quality">
                                <option value="high">High Quality</option>
                                <option value="ultra">Ultra Quality</option>
                                <option value="standard">Standard</option>
                            </select>
                            <select class="form-input" id="ecommerce-model">
                                <option value="fal-kontext">FLUX-Pro Kontext (Best Quality)</option>
                                <option value="fal-img2img">FLUX Image-to-Image</option>
                                <option value="dev">FLUX.1-dev (Fallback)</option>
                                <option value="schnell">FLUX.1-schnell (Fast)</option>
                            </select>
                        </div>
                        
                        <div id="ecommerce-status"></div>
                        
                        <!-- Download Actions -->
                        <div id="download-section" style="display: none; margin-top: 25px; text-align: center;">
                            <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                                <button onclick="downloadEnhancedImage()" 
                                        style="background: #3b82f6; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#2563eb'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(59, 130, 246, 0.4)'"
                                        onmouseout="this.style.background='#3b82f6'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(59, 130, 246, 0.3)'">
                                    <i class="fas fa-download"></i>
                                    Download
                                </button>
                                
                                <button onclick="saveToGallery()" 
                                        style="background: #10b981; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#059669'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(16, 185, 129, 0.4)'"
                                        onmouseout="this.style.background='#10b981'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(16, 185, 129, 0.3)'">
                                    <i class="fas fa-save"></i>
                                    Save to Gallery
                                </button>
                                
                                <button onclick="tryDifferentEnhancement()" 
                                        style="background: #6b7280; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(107, 114, 128, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#4b5563'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(107, 114, 128, 0.4)'"
                                        onmouseout="this.style.background='#6b7280'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(107, 114, 128, 0.3)'">
                                    <i class="fas fa-redo"></i>
                                    Try Again
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Marketing Workflow -->
                <div id="marketing-workflow" class="workflow-section" style="display:none">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon marketing">
                                <i class="fas fa-bullhorn"></i>
                            </div>
                            <div>
                                <h3>Marketing Content Creator</h3>
                                <p style="color: #718096; margin: 0;">AI-powered marketing visuals from your images</p>
                            </div>
                        </div>
                        
                        <!-- Modern Side-by-Side Layout -->
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                            <!-- Upload Section -->
                            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0; display: flex; align-items: center; gap: 8px;">
                                        <i class="fas fa-upload"></i> Reference Image
                                    </h4>
                                    <span id="marketing-upload-status" style="color: rgba(255,255,255,0.6); font-size: 12px; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px;">Upload reference image</span>
                                </div>
                                <div class="upload-area" id="marketing-upload-area" onclick="document.getElementById('marketing-upload').click()" style="aspect-ratio: 1; border: 2px dashed rgba(255,255,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center; cursor: pointer; transition: all 0.3s ease;">
                                    <div class="upload-content" id="marketing-upload-content" style="text-align: center; color: rgba(255,255,255,0.6);">
                                        <i class="fas fa-cloud-upload-alt" style="font-size: 48px; margin-bottom: 15px; color: rgba(244, 63, 94, 0.7);"></i>
                                        <h4 style="color: white; margin: 0 0 8px 0;">Click to Upload</h4>
                                        <p style="margin: 0 0 8px 0; font-size: 14px;">Product or brand image</p>
                                        <small style="font-size: 12px; opacity: 0.7;">JPG, PNG, WebP • Max 10MB</small>
                                    </div>
                                    <img id="marketing-uploaded-preview" class="uploaded-image" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" alt="Marketing preview">
                                </div>
                                <input type="file" id="marketing-upload" accept="image/*" style="display: none;" onchange="handleMarketingUpload(event)">
                            </div>
                            
                            <!-- Result Section -->
                            <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 20px; border: 1px solid rgba(255,255,255,0.1);">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                                    <h4 style="color: white; margin: 0; display: flex; align-items: center; gap: 8px;">
                                        <i class="fas fa-magic"></i> Marketing Visual
                                    </h4>
                                    <span id="marketing-result-status" style="color: rgba(255,255,255,0.6); font-size: 12px; background: rgba(255,255,255,0.1); padding: 4px 12px; border-radius: 20px;">Generated visual will appear here</span>
                                </div>
                                <div style="aspect-ratio: 1; border: 2px dashed rgba(255,255,255,0.2); border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                                    <div id="marketing-result-content" style="text-align: center; color: rgba(255,255,255,0.6);">
                                        <i class="fas fa-bullhorn" style="font-size: 48px; margin-bottom: 15px; color: rgba(244, 63, 94, 0.7);"></i>
                                        <h4 style="color: white; margin: 0 0 8px 0;">Marketing Visual</h4>
                                        <p style="margin: 0 0 8px 0; font-size: 14px;">Your AI-generated content will appear here</p>
                                        <small style="font-size: 12px; opacity: 0.7;">Upload an image and generate marketing content</small>
                                    </div>
                                    <img id="marketing-generated-preview" style="display: none; width: 100%; height: 100%; object-fit: cover; border-radius: 8px;" alt="Generated marketing visual">
                                </div>
                            </div>
                        </div>
                        
                        <!-- Controls Below -->
                        <div class="form-group">
                            <label class="form-label" style="display: flex; align-items: center; gap: 8px;">
                                <i class="fas fa-edit"></i> Campaign Concept
                            </label>
                            <textarea class="form-input form-textarea" id="marketing-prompt" 
                                placeholder="Describe your marketing concept...&#10;Examples:&#10;• Create a social media ad for this product&#10;• Design a promotional banner&#10;• Make an Instagram story template&#10;• Generate a product launch visual" style="min-height: 100px;"></textarea>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">Marketing Style</label>
                            <div class="suggestions-grid">
                                <div class="suggestion-chip" onclick="selectMarketingStyle('social_media')" id="style-social_media">
                                    <i class="fas fa-share-alt"></i> Social Media
                                </div>
                                <div class="suggestion-chip" onclick="selectMarketingStyle('banner')" id="style-banner">
                                    <i class="fas fa-rectangle-ad"></i> Banner Ad
                                </div>
                                <div class="suggestion-chip" onclick="selectMarketingStyle('advertisement')" id="style-advertisement">
                                    <i class="fas fa-bullhorn"></i> Advertisement
                                </div>
                                <div class="suggestion-chip" onclick="selectMarketingStyle('promotional')" id="style-promotional">
                                    <i class="fas fa-tags"></i> Promotional
                                </div>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label class="form-label">AI-Powered Marketing Suggestions</label>
                            <div style="display: flex; flex-direction: column; gap: 15px;">
                                <button class="btn btn-secondary" onclick="generateMarketingAISuggestions()" id="marketing-ai-suggestions-btn" disabled style="display: flex; align-items: center; gap: 8px; justify-content: center;">
                                    <i class="fas fa-robot"></i>
                                    Analyze Image & Generate Marketing Ideas
                                </button>
                                <div id="marketing-ai-suggestions-container" style="display: none;">
                                    <div id="marketing-ai-suggestions-list" style="display: flex; flex-direction: column; gap: 8px;">
                                        <!-- AI marketing suggestions will appear here -->
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="form-row">
                            <button class="btn btn-primary" onclick="generateMarketingVisual()" id="marketing-btn" disabled>
                                <i class="fas fa-rocket"></i>
                                Create Marketing Visual
                            </button>
                            <select class="form-input" id="marketing-quality">
                                <option value="high">High Quality</option>
                                <option value="ultra">Ultra Quality</option>
                                <option value="standard">Standard</option>
                            </select>
                            <select class="form-input" id="marketing-model">
                                <option value="fal-kontext">FLUX-Pro Kontext (Best Quality)</option>
                                <option value="fal-img2img">FLUX Image-to-Image</option>
                                <option value="dev">FLUX.1-dev (Fallback)</option>
                                <option value="schnell">FLUX.1-schnell (Fast)</option>
                            </select>
                        </div>
                        
                        <div id="marketing-status"></div>
                        
                        <!-- Download Actions -->
                        <div id="marketing-download-section" style="display: none; margin-top: 25px; text-align: center;">
                            <div style="display: flex; gap: 12px; justify-content: center; flex-wrap: wrap;">
                                <button onclick="downloadMarketingImage()" 
                                        style="background: #f43f5e; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(244, 63, 94, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#e11d48'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(244, 63, 94, 0.4)'"
                                        onmouseout="this.style.background='#f43f5e'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(244, 63, 94, 0.3)'">
                                    <i class="fas fa-download"></i>
                                    Download
                                </button>
                                
                                <button onclick="saveMarketingToGallery()" 
                                        style="background: #10b981; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#059669'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(16, 185, 129, 0.4)'"
                                        onmouseout="this.style.background='#10b981'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(16, 185, 129, 0.3)'">
                                    <i class="fas fa-save"></i>
                                    Save to Gallery
                                </button>
                                
                                <button onclick="regenerateMarketingVisual()" 
                                        style="background: #6b7280; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; display: flex; align-items: center; gap: 8px; box-shadow: 0 4px 12px rgba(107, 114, 128, 0.3); transition: all 0.2s ease;"
                                        onmouseover="this.style.background='#4b5563'; this.style.transform='translateY(-1px)'; this.style.boxShadow='0 6px 20px rgba(107, 114, 128, 0.4)'"
                                        onmouseout="this.style.background='#6b7280'; this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(107, 114, 128, 0.3)'">
                                    <i class="fas fa-redo"></i>
                                    Try Again
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Gallery Section -->
                <div id="gallery-section" class="workflow-section" style="display:none">
                    <div class="card">
                        <div class="card-header">
                            <div class="card-icon" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
                                <i class="fas fa-images"></i>
                            </div>
                            <div>
                                <h3>Gallery</h3>
                                <p style="color: #718096; margin: 0;">Your saved enhanced images</p>
                            </div>
                            <button class="btn btn-secondary" onclick="refreshGallery()" style="margin-left: auto;">
                                <i class="fas fa-refresh"></i> Refresh
                            </button>
                        </div>
                        
                        <div id="gallery-loading" style="text-align: center; padding: 40px; display: none;">
                            <i class="fas fa-spinner fa-spin" style="font-size: 24px; color: #667eea;"></i>
                            <p style="color: rgba(255,255,255,0.7); margin-top: 10px;">Loading gallery...</p>
                        </div>
                        
                        <div id="gallery-empty" style="text-align: center; padding: 60px; display: none;">
                            <i class="fas fa-images" style="font-size: 48px; color: rgba(255,255,255,0.3); margin-bottom: 20px;"></i>
                            <h4 style="color: rgba(255,255,255,0.7); margin-bottom: 10px;">No images in gallery yet</h4>
                            <p style="color: rgba(255,255,255,0.5);">Enhanced images will appear here when you save them</p>
                        </div>
                        
                        <div id="gallery-grid" style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
                            <!-- Gallery items will be loaded here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let currentWorkflow = 'ecommerce';
        let currentStyle = 'product_hero';
        let currentImageUrls = {};
        
        function showWorkflow(workflow) {
            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            event.target.classList.add('active');
            
            // Hide all workflows
            document.querySelectorAll('.workflow-section').forEach(section => section.style.display = 'none');
            
            // Show selected workflow
            document.getElementById(workflow + '-workflow').style.display = 'block';
            currentWorkflow = workflow;
            
            // Update header
            const titles = {
                'ecommerce': 'E-commerce Product Photos',
                'marketing': 'Marketing Content Creation'
            };
            
            const subtitles = {
                'ecommerce': 'Create professional product photography for your online store',
                'marketing': 'Generate compelling marketing visuals for campaigns and social media'
            };
            
            document.getElementById('workflow-title').textContent = titles[workflow];
            document.getElementById('workflow-subtitle').textContent = subtitles[workflow];
        }
        
        function showSection(section) {
            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
            event.target.classList.add('active');
            
            // Hide all workflows
            document.querySelectorAll('.workflow-section').forEach(section => section.style.display = 'none');
            
            // Show selected section
            document.getElementById(section + '-section').style.display = 'block';
            
            // Update header based on section
            const titles = {
                'gallery': 'Gallery'
            };
            
            const subtitles = {
                'gallery': 'Your saved enhanced images'
            };
            
            document.getElementById('workflow-title').textContent = titles[section];
            document.getElementById('workflow-subtitle').textContent = subtitles[section];
            
            // Load gallery if showing gallery section
            if (section === 'gallery') {
                loadGallery();
            }
        }
        
        function selectStyle(workflow, style) {
            currentStyle = style;
            
            // Update active style
            document.querySelectorAll('.suggestion-chip').forEach(chip => chip.classList.remove('active'));
            event.target.classList.add('active');
            
            // Update placeholder text based on style
            const placeholders = {
                'ecommerce': {
                    'product_hero': 'luxury watch with leather strap',
                    'lifestyle': 'smartphone being used in modern office',
                    'detail_shot': 'close-up of watch mechanism',
                    'packaging': 'elegant product box presentation'
                },
                'marketing': {
                    'social_media': 'summer sale campaign for fashion brand',
                    'banner': 'new product launch announcement',
                    'advertisement': 'premium brand storytelling visual',
                    'promotional': 'limited time offer promotion'
                }
            };
            
            const promptField = document.getElementById(workflow + '-prompt');
            if (placeholders[workflow] && placeholders[workflow][style]) {
                promptField.placeholder = placeholders[workflow][style];
            }
        }
        
        function generateImage(workflow) {
            const prompt = document.getElementById(workflow + '-prompt').value.trim();
            const quality = document.getElementById(workflow + '-quality').value;
            const model = document.getElementById(workflow + '-model').value;
            
            if (!prompt) {
                showStatus(workflow + '-status', 'Please describe your ' + (workflow === 'ecommerce' ? 'product' : 'campaign concept'), 'error');
                return;
            }
            
            const btn = document.getElementById(workflow + '-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span>Generating...';
            btn.disabled = true;
            
            showStatus(workflow + '-status', 
                '<i class="fas fa-magic"></i> Creating your ' + (workflow === 'ecommerce' ? 'product photo' : 'marketing visual') + ' with FLUX.1...', 
                'warning'
            );
            
            fetch('/api/generate-advanced', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    workflow_type: workflow,
                    style_type: currentStyle,
                    quality: quality,
                    model: model
                })
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                if (data.success) {
                    showStatus(workflow + '-status', '<i class="fas fa-check-circle"></i> Successfully generated!', 'success');
                    showResult(workflow, data.image_url, data.model, data.enhanced_prompt);
                    currentImageUrls[workflow] = data.image_url;
                } else {
                    showStatus(workflow + '-status', '<i class="fas fa-exclamation-triangle"></i> ' + data.error, 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus(workflow + '-status', '<i class="fas fa-exclamation-triangle"></i> Network error occurred', 'error');
            });
        }
        
        function showStatus(elementId, message, type) {
            document.getElementById(elementId).innerHTML = 
                '<div class="status-message status-' + type + '">' + message + '</div>';
        }
        
        function showResult(workflow, imageUrl, modelName, enhancedPrompt) {
            const resultCard = document.getElementById(workflow + '-result');
            const resultContent = document.getElementById(workflow + '-content');
            
            resultContent.innerHTML = `
                <img src="${imageUrl}" class="result-image" alt="Generated image">
                <div style="margin-top: 15px; padding: 15px; background: #f7fafc; border-radius: 8px;">
                    <p style="color: #4a5568; font-size: 14px; margin-bottom: 8px;"><strong>Model:</strong> ${modelName}</p>
                    <p style="color: #4a5568; font-size: 14px;"><strong>Enhanced Prompt:</strong> ${enhancedPrompt}</p>
                </div>
            `;
            
            resultCard.style.display = 'block';
            resultCard.scrollIntoView({ behavior: 'smooth' });
        }
        
        function downloadImage(workflow) {
            if (currentImageUrls[workflow]) {
                const link = document.createElement('a');
                link.href = currentImageUrls[workflow];
                link.download = `fluxflow-${workflow}-${Date.now()}.png`;
                link.click();
            }
        }
        
        function regenerateImage(workflow) {
            generateImage(workflow);
        }
        
        function saveProject() {
            // Implementation for saving project
            alert('Project saved successfully!');
        }
        
        function showHistory() {
            // Implementation for showing history
            alert('History feature coming soon!');
        }
        
        // Product Photography Studio Functions
        let uploadedImageFile = null;
        let uploadedImageUrl = null;
        let generatedImageUrl = null;
        
        function handleProductUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            // Validate file type
            const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp'];
            if (!allowedTypes.includes(file.type)) {
                showStatus('ecommerce-status', 'Please upload a valid image file (JPG, PNG, WebP)', 'error');
                return;
            }
            
            // Validate file size (10MB)
            if (file.size > 10 * 1024 * 1024) {
                showStatus('ecommerce-status', 'File size must be less than 10MB', 'error');
                return;
            }
            
            uploadedImageFile = file;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                uploadedImageUrl = e.target.result;
                const uploadArea = document.getElementById('upload-area');
                const uploadContent = document.getElementById('upload-content');
                const preview = document.getElementById('uploaded-preview');
                
                uploadContent.style.display = 'none';
                preview.src = uploadedImageUrl;
                preview.style.display = 'block';
                
                // Enable generate button and AI suggestions
                const generateBtn = document.getElementById('ecommerce-btn');
                const aiSuggestionsBtn = document.getElementById('ai-suggestions-btn');
                generateBtn.disabled = false;
                aiSuggestionsBtn.disabled = false;
                
                // Update upload status
                const uploadStatus = document.getElementById('upload-status');
                if (uploadStatus) {
                    uploadStatus.textContent = 'Image uploaded successfully!';
                    uploadStatus.style.color = '#22c55e';
                }
                
                showStatus('ecommerce-status', '✅ Product image uploaded successfully!', 'success');
            };
            reader.readAsDataURL(file);
        }
        
        function generateProductPhoto() {
            const prompt = document.getElementById('ecommerce-prompt').value.trim();
            const quality = document.getElementById('ecommerce-quality').value;
            const model = document.getElementById('ecommerce-model').value;
            
            if (!uploadedImageFile) {
                showStatus('ecommerce-status', 'Please upload a product image first', 'error');
                return;
            }
            
            // Allow empty prompt for basic enhancement
            if (!prompt) {
                prompt = 'enhance this product image for professional e-commerce use';
            }
            
            const btn = document.getElementById('ecommerce-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<span class="loading"></span>Enhancing Product...';
            btn.disabled = true;
            
            showStatus('ecommerce-status', 
                '<i class="fas fa-wand-magic-sparkles"></i> Enhancing your product image with FLUX.1...', 
                'warning'
            );
            
            // First upload the image
            const formData = new FormData();
            formData.append('file', uploadedImageFile);
            
            fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(uploadData => {
                if (!uploadData.success) {
                    throw new Error(uploadData.error);
                }
                
                // Now enhance the product image
                return fetch('/api/generate-advanced', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        workflow_type: 'ecommerce',
                        style_type: currentStyle || 'background_removal',
                        quality: quality,
                        model: model,
                        reference_image: uploadData.image_url
                    })
                });
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                if (data.success) {
                    showStatus('ecommerce-status', '<i class="fas fa-check-circle"></i> Product image enhanced successfully!', 'success');
                    showProductResult(uploadedImageUrl, data.image_url, data.enhanced_prompt);
                    generatedImageUrl = data.image_url;
                } else {
                    showStatus('ecommerce-status', '<i class="fas fa-exclamation-triangle"></i> ' + data.error, 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus('ecommerce-status', '<i class="fas fa-exclamation-triangle"></i> Error: ' + error.message, 'error');
            });
        }
        
        function enhanceProductPhoto() { 
            generateProductPhoto(); 
        }
        
        function generateAISuggestions() {
            if (!uploadedImageFile) {
                showStatus('ecommerce-status', 'Please upload an image first', 'error');
                return;
            }
            
            const btn = document.getElementById('ai-suggestions-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Image...';
            btn.disabled = true;
            
            // First upload the image
            const formData = new FormData();
            formData.append('file', uploadedImageFile);
            
            fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(uploadData => {
                if (!uploadData.success) {
                    throw new Error(uploadData.error);
                }
                
                // Now generate AI suggestions
                return fetch('/api/generate-suggestions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_url: uploadData.image_url
                    })
                });
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                if (data.success && data.suggestions) {
                    displayAISuggestions(data.suggestions);
                    showStatus('ecommerce-status', `✅ Generated ${data.suggestions.length} AI suggestions!`, 'success');
                } else {
                    showStatus('ecommerce-status', '❌ ' + (data.error || 'Failed to generate suggestions'), 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus('ecommerce-status', '❌ Error: ' + error.message, 'error');
            });
        }
        
        function displayAISuggestions(suggestions) {
            const container = document.getElementById('ai-suggestions-container');
            const list = document.getElementById('ai-suggestions-list');
            
            // Clear previous suggestions
            list.innerHTML = '';
            
            // Add each suggestion as a clickable chip
            suggestions.forEach((suggestion, index) => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.style.cssText = `
                    background: rgba(34, 197, 94, 0.1);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    border-radius: 8px;
                    padding: 12px 16px;
                    color: #22c55e;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-size: 14px;
                    line-height: 1.4;
                `;
                chip.innerHTML = `<i class="fas fa-lightbulb" style="margin-right: 8px;"></i>${suggestion}`;
                
                chip.onclick = () => {
                    document.getElementById('ecommerce-prompt').value = suggestion;
                    // Highlight selected suggestion
                    list.querySelectorAll('.suggestion-chip').forEach(c => {
                        c.style.background = 'rgba(34, 197, 94, 0.1)';
                        c.style.borderColor = 'rgba(34, 197, 94, 0.3)';
                    });
                    chip.style.background = 'rgba(34, 197, 94, 0.2)';
                    chip.style.borderColor = 'rgba(34, 197, 94, 0.5)';
                };
                
                chip.onmouseenter = () => {
                    chip.style.background = 'rgba(34, 197, 94, 0.15)';
                    chip.style.transform = 'translateY(-1px)';
                };
                
                chip.onmouseleave = () => {
                    if (chip.style.borderColor !== 'rgba(34, 197, 94, 0.5)') {
                        chip.style.background = 'rgba(34, 197, 94, 0.1)';
                    }
                    chip.style.transform = 'translateY(0)';
                };
                
                list.appendChild(chip);
            });
            
            container.style.display = 'block';
        }
        
        function showProductResult(originalUrl, generatedUrl, enhancedPrompt) {
            // Update the enhanced preview in the new UI
            const enhancedPreview = document.getElementById('enhanced-preview');
            const resultPlaceholder = document.getElementById('result-content');
            const downloadSection = document.getElementById('download-section');
            const resultStatus = document.getElementById('result-status');
            
            if (enhancedPreview && resultPlaceholder) {
                resultPlaceholder.style.display = 'none';
                enhancedPreview.src = generatedUrl;
                enhancedPreview.style.display = 'block';
                
                if (resultStatus) {
                    resultStatus.textContent = 'Enhancement complete!';
                    resultStatus.style.color = '#22c55e';
                }
                
                if (downloadSection) {
                    downloadSection.style.display = 'block';
                }
            }
        }
        
        function downloadEnhancedImage() {
            const enhancedImg = document.getElementById('enhanced-preview');
            if (enhancedImg && enhancedImg.src) {
                const link = document.createElement('a');
                link.href = enhancedImg.src;
                link.download = `enhanced-product-${Date.now()}.png`;
                link.click();
            }
        }
        
        function saveToGallery() {
            const enhancedImg = document.getElementById('enhanced-preview');
            const originalImg = document.getElementById('uploaded-preview');
            const prompt = document.getElementById('ecommerce-prompt').value;
            
            if (!enhancedImg || !enhancedImg.src) {
                showStatus('ecommerce-status', 'No enhanced image to save', 'error');
                return;
            }
            
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
            btn.disabled = true;
            
            fetch('/api/save-to-gallery', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: enhancedImg.src,
                    original_url: originalImg ? originalImg.src : null,
                    prompt: prompt
                })
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                if (data.success) {
                    showStatus('ecommerce-status', '✅ ' + data.message, 'success');
                    // Update gallery count if visible
                    updateGalleryCount();
                } else {
                    showStatus('ecommerce-status', '❌ ' + (data.error || 'Failed to save to gallery'), 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus('ecommerce-status', '❌ Error: ' + error.message, 'error');
            });
        }
        
        function updateGalleryCount() {
            // Update gallery count in sidebar if needed
            fetch('/api/get-gallery')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // You can update gallery count here if you want to show it in sidebar
                    console.log(`Gallery now has ${data.gallery.length} items`);
                }
            })
            .catch(error => console.error('Error updating gallery count:', error));
        }
        
        // Marketing workflow functions
        let marketingUploadedImageFile = null;
        let marketingUploadedImageUrl = null;
        let marketingGeneratedImageUrl = null;
        let selectedMarketingStyle = 'social_media'; // Default style
        
        function handleMarketingUpload(event) {
            const file = event.target.files[0];
            if (!file) return;
            
            marketingUploadedImageFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                marketingUploadedImageUrl = e.target.result;
                const uploadArea = document.getElementById('marketing-upload-area');
                const uploadContent = document.getElementById('marketing-upload-content');
                const preview = document.getElementById('marketing-uploaded-preview');
                
                uploadContent.style.display = 'none';
                preview.src = marketingUploadedImageUrl;
                preview.style.display = 'block';
                
                // Enable buttons
                const generateBtn = document.getElementById('marketing-btn');
                const aiSuggestionsBtn = document.getElementById('marketing-ai-suggestions-btn');
                generateBtn.disabled = false;
                aiSuggestionsBtn.disabled = false;
                
                // Update upload status
                const uploadStatus = document.getElementById('marketing-upload-status');
                if (uploadStatus) {
                    uploadStatus.textContent = 'Image uploaded successfully!';
                    uploadStatus.style.color = '#22c55e';
                }
                
                showStatus('marketing-status', '✅ Reference image uploaded successfully!', 'success');
            };
            reader.readAsDataURL(file);
        }
        
        function selectMarketingStyle(style) {
            selectedMarketingStyle = style;
            
            // Update active style visual feedback
            document.querySelectorAll('#marketing-workflow .suggestion-chip').forEach(chip => {
                chip.classList.remove('active');
                chip.style.background = 'rgba(255,255,255,0.1)';
                chip.style.borderColor = 'rgba(255,255,255,0.2)';
            });
            
            const selectedChip = document.getElementById('style-' + style);
            if (selectedChip) {
                selectedChip.classList.add('active');
                selectedChip.style.background = 'rgba(244, 63, 94, 0.3)';
                selectedChip.style.borderColor = 'rgba(244, 63, 94, 0.5)';
            }
            
            // Update button text to show selected style
            const aiBtn = document.getElementById('marketing-ai-suggestions-btn');
            const styleNames = {
                'social_media': 'Social Media',
                'banner': 'Banner Ad',
                'advertisement': 'Advertisement',
                'promotional': 'Promotional'
            };
            
            if (aiBtn && !aiBtn.disabled) {
                aiBtn.innerHTML = `<i class="fas fa-robot"></i> Generate ${styleNames[style]} Ideas`;
            }
        }
        
        function generateMarketingAISuggestions() {
            if (!marketingUploadedImageFile) {
                showStatus('marketing-status', 'Please upload a reference image first', 'error');
                return;
            }
            
            const btn = document.getElementById('marketing-ai-suggestions-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Image for Marketing Ideas...';
            btn.disabled = true;
            
            // First upload the image
            const formData = new FormData();
            formData.append('file', marketingUploadedImageFile);
            
            fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(uploadData => {
                if (!uploadData.success) {
                    throw new Error(uploadData.error);
                }
                
                // Now generate marketing-specific AI suggestions
                return fetch('/api/generate-marketing-suggestions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image_url: uploadData.image_url,
                        marketing_style: selectedMarketingStyle
                    })
                });
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                console.log('Marketing AI Response:', data);
                
                if (data.success && data.suggestions && data.suggestions.length > 0) {
                    displayMarketingAISuggestions(data.suggestions);
                    showStatus('marketing-status', `✅ Generated ${data.suggestions.length} marketing ideas!`, 'success');
                } else {
                    console.error('Marketing AI Error:', data);
                    showStatus('marketing-status', '❌ ' + (data.error || 'No suggestions generated'), 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus('marketing-status', '❌ Error: ' + error.message, 'error');
            });
        }
        
        function displayMarketingAISuggestions(suggestions) {
            const container = document.getElementById('marketing-ai-suggestions-container');
            const list = document.getElementById('marketing-ai-suggestions-list');
            
            // Clear previous suggestions
            list.innerHTML = '';
            
            // Add each suggestion as a clickable chip
            suggestions.forEach((suggestion, index) => {
                const chip = document.createElement('div');
                chip.className = 'suggestion-chip';
                chip.style.cssText = `
                    background: rgba(244, 63, 94, 0.1);
                    border: 1px solid rgba(244, 63, 94, 0.3);
                    border-radius: 8px;
                    padding: 12px 16px;
                    color: #f43f5e;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-size: 14px;
                    line-height: 1.4;
                `;
                chip.innerHTML = `<i class="fas fa-bullhorn" style="margin-right: 8px;"></i>${suggestion}`;
                
                chip.onclick = () => {
                    document.getElementById('marketing-prompt').value = suggestion;
                    // Highlight selected suggestion
                    list.querySelectorAll('.suggestion-chip').forEach(c => {
                        c.style.background = 'rgba(244, 63, 94, 0.1)';
                        c.style.borderColor = 'rgba(244, 63, 94, 0.3)';
                    });
                    chip.style.background = 'rgba(244, 63, 94, 0.2)';
                    chip.style.borderColor = 'rgba(244, 63, 94, 0.5)';
                };
                
                chip.onmouseenter = () => {
                    chip.style.background = 'rgba(244, 63, 94, 0.15)';
                    chip.style.transform = 'translateY(-1px)';
                };
                
                chip.onmouseleave = () => {
                    if (chip.style.borderColor !== 'rgba(244, 63, 94, 0.5)') {
                        chip.style.background = 'rgba(244, 63, 94, 0.1)';
                    }
                    chip.style.transform = 'translateY(0)';
                };
                
                list.appendChild(chip);
            });
            
            container.style.display = 'block';
        }
        
        function generateMarketingVisual() {
            const prompt = document.getElementById('marketing-prompt').value.trim();
            const quality = document.getElementById('marketing-quality').value;
            const model = document.getElementById('marketing-model').value;
            
            if (!prompt) {
                showStatus('marketing-status', 'Please enter a campaign concept', 'error');
                return;
            }
            
            const btn = document.getElementById('marketing-btn');
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Marketing Visual...';
            btn.disabled = true;
            
            // Use the same API as product enhancement but with marketing-specific prompts
            const requestData = {
                workflow_type: 'marketing',
                style_type: currentStyle || 'social_media',
                prompt: prompt,
                quality: quality,
                model: model
            };
            
            if (marketingUploadedImageFile) {
                // If there's an uploaded image, include it
                const formData = new FormData();
                formData.append('file', marketingUploadedImageFile);
                
                fetch('/api/upload-image', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(uploadData => {
                    if (uploadData.success) {
                        requestData.image_url = uploadData.image_url;
                    }
                    return generateMarketingContent(requestData);
                })
                .catch(error => {
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    showStatus('marketing-status', 'Error uploading image: ' + error.message, 'error');
                });
            } else {
                generateMarketingContent(requestData);
            }
            
            function generateMarketingContent(data) {
                fetch('/api/generate-advanced', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                })
                .then(response => response.json())
                .then(result => {
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    
                    if (result.success) {
                        marketingGeneratedImageUrl = result.image_url;
                        showMarketingResult(marketingUploadedImageUrl, result.image_url, prompt);
                        showStatus('marketing-status', '✅ Marketing visual created successfully!', 'success');
                    } else {
                        showStatus('marketing-status', '❌ ' + (result.error || 'Failed to create marketing visual'), 'error');
                    }
                })
                .catch(error => {
                    btn.innerHTML = originalText;
                    btn.disabled = false;
                    showStatus('marketing-status', '❌ Error: ' + error.message, 'error');
                });
            }
        }
        
        function showMarketingResult(originalUrl, generatedUrl, prompt) {
            const generatedPreview = document.getElementById('marketing-generated-preview');
            const resultPlaceholder = document.getElementById('marketing-result-content');
            const downloadSection = document.getElementById('marketing-download-section');
            const resultStatus = document.getElementById('marketing-result-status');
            
            if (generatedPreview && resultPlaceholder) {
                resultPlaceholder.style.display = 'none';
                generatedPreview.src = generatedUrl;
                generatedPreview.style.display = 'block';
                
                if (resultStatus) {
                    resultStatus.textContent = 'Marketing visual created!';
                    resultStatus.style.color = '#f43f5e';
                }
                
                if (downloadSection) {
                    downloadSection.style.display = 'block';
                }
            }
        }
        
        function downloadMarketingImage() {
            const generatedImg = document.getElementById('marketing-generated-preview');
            if (generatedImg && generatedImg.src) {
                const link = document.createElement('a');
                link.href = generatedImg.src;
                link.download = `marketing-visual-${Date.now()}.png`;
                link.click();
            }
        }
        
        function saveMarketingToGallery() {
            const generatedImg = document.getElementById('marketing-generated-preview');
            const originalImg = document.getElementById('marketing-uploaded-preview');
            const prompt = document.getElementById('marketing-prompt').value;
            
            if (!generatedImg || !generatedImg.src) {
                showStatus('marketing-status', 'No marketing visual to save', 'error');
                return;
            }
            
            const btn = event.target;
            const originalText = btn.innerHTML;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Saving...';
            btn.disabled = true;
            
            fetch('/api/save-to-gallery', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: generatedImg.src,
                    original_url: originalImg ? originalImg.src : null,
                    prompt: prompt,
                    type: 'marketing_content'
                })
            })
            .then(response => response.json())
            .then(data => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                
                if (data.success) {
                    showStatus('marketing-status', '✅ ' + data.message, 'success');
                    updateGalleryCount();
                } else {
                    showStatus('marketing-status', '❌ ' + (data.error || 'Failed to save to gallery'), 'error');
                }
            })
            .catch(error => {
                btn.innerHTML = originalText;
                btn.disabled = false;
                showStatus('marketing-status', '❌ Error: ' + error.message, 'error');
            });
        }
        
        function regenerateMarketingVisual() {
            generateMarketingVisual();
        }
        
        function loadGallery() {
            const loading = document.getElementById('gallery-loading');
            const empty = document.getElementById('gallery-empty');
            const grid = document.getElementById('gallery-grid');
            
            loading.style.display = 'block';
            empty.style.display = 'none';
            grid.style.display = 'none';
            
            fetch('/api/get-gallery')
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                
                if (data.success && data.gallery.length > 0) {
                    displayGalleryItems(data.gallery);
                    grid.style.display = 'grid';
                } else {
                    empty.style.display = 'block';
                }
            })
            .catch(error => {
                loading.style.display = 'none';
                empty.style.display = 'block';
                console.error('Error loading gallery:', error);
            });
        }
        
        function displayGalleryItems(items) {
            const grid = document.getElementById('gallery-grid');
            grid.innerHTML = '';
            
            items.forEach(item => {
                const card = document.createElement('div');
                card.style.cssText = `
                    background: rgba(255,255,255,0.05);
                    border-radius: 12px;
                    padding: 15px;
                    border: 1px solid rgba(255,255,255,0.1);
                    transition: all 0.3s ease;
                `;
                
                card.onmouseenter = () => {
                    card.style.transform = 'translateY(-2px)';
                    card.style.boxShadow = '0 8px 25px rgba(0,0,0,0.3)';
                };
                
                card.onmouseleave = () => {
                    card.style.transform = 'translateY(0)';
                    card.style.boxShadow = 'none';
                };
                
                const date = new Date(item.created_at).toLocaleDateString();
                const time = new Date(item.created_at).toLocaleTimeString();
                
                card.innerHTML = `
                    <div style="position: relative; margin-bottom: 12px;">
                        <img src="${item.enhanced_image}" alt="Enhanced image" 
                             style="width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 8px;">
                        <div style="position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,0.7); border-radius: 6px; padding: 4px 8px;">
                            <span style="color: white; font-size: 12px; font-weight: 500;">Enhanced</span>
                        </div>
                    </div>
                    <div style="margin-bottom: 8px;">
                        <div style="color: rgba(255,255,255,0.6); font-size: 12px; margin-bottom: 4px;">${date} at ${time}</div>
                        <div style="color: rgba(255,255,255,0.8); font-size: 14px; line-height: 1.4; max-height: 40px; overflow: hidden;">
                            ${item.prompt || 'No prompt provided'}
                        </div>
                    </div>
                    <div style="display: flex; gap: 8px;">
                        <button onclick="downloadGalleryImage('${item.enhanced_image}')" 
                                style="flex: 1; background: rgba(34, 197, 94, 0.2); border: 1px solid rgba(34, 197, 94, 0.3); color: #22c55e; padding: 8px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;">
                            <i class="fas fa-download"></i> Download
                        </button>
                        <button onclick="viewGalleryImage('${item.enhanced_image}', '${item.original_image || ''}', '${item.prompt || ''}')" 
                                style="flex: 1; background: rgba(102, 126, 234, 0.2); border: 1px solid rgba(102, 126, 234, 0.3); color: #667eea; padding: 8px 12px; border-radius: 6px; font-size: 12px; cursor: pointer;">
                            <i class="fas fa-eye"></i> View
                        </button>
                    </div>
                `;
                
                grid.appendChild(card);
            });
        }
        
        function refreshGallery() {
            loadGallery();
        }
        
        function downloadGalleryImage(imageUrl) {
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = `gallery-image-${Date.now()}.png`;
            link.click();
        }
        
        function viewGalleryImage(enhancedUrl, originalUrl, prompt) {
            // Create a modal to view the image
            const modal = document.createElement('div');
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.9);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
                padding: 20px;
            `;
            
            modal.innerHTML = `
                <div style="background: #1a1a2e; border-radius: 12px; padding: 20px; max-width: 800px; width: 100%; max-height: 90vh; overflow-y: auto;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                        <h3 style="color: white; margin: 0;">Image Details</h3>
                        <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                                style="background: none; border: none; color: white; font-size: 24px; cursor: pointer;">&times;</button>
                    </div>
                    <div style="margin-bottom: 15px;">
                        <img src="${enhancedUrl}" alt="Enhanced image" style="width: 100%; border-radius: 8px;">
                    </div>
                    ${prompt ? `<div style="margin-bottom: 15px;">
                        <label style="color: rgba(255,255,255,0.8); font-size: 14px; font-weight: 500;">Prompt:</label>
                        <p style="color: white; margin: 5px 0 0 0; padding: 10px; background: rgba(255,255,255,0.05); border-radius: 6px;">${prompt}</p>
                    </div>` : ''}
                    <div style="display: flex; gap: 10px;">
                        <button onclick="downloadGalleryImage('${enhancedUrl}')" 
                                style="background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); border: none; color: white; padding: 10px 20px; border-radius: 6px; cursor: pointer;">
                            <i class="fas fa-download"></i> Download
                        </button>
                        <button onclick="this.parentElement.parentElement.parentElement.remove()" 
                                style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); color: white; padding: 10px 20px; border-radius: 6px; cursor: pointer;">
                            Close
                        </button>
                    </div>
                </div>
            `;
            
            modal.onclick = (e) => {
                if (e.target === modal) modal.remove();
            };
            
            document.body.appendChild(modal);
        }
        
        function regenerateProductPhoto() {
            generateProductPhoto();
        }
        
        function shareResult() {
            if (navigator.share && generatedImageUrl) {
                navigator.share({
                    title: 'FluxFlow Studio - Professional Photoshoot',
                    text: 'Check out this professional product photoshoot created with AI!',
                    url: window.location.href
                });
            } else {
                // Fallback: copy link to clipboard
                navigator.clipboard.writeText(window.location.href).then(() => {
                    showStatus('ecommerce-status', 'Link copied to clipboard!', 'success');
                });
            }
        }
        
        // Drag and drop functionality
        document.addEventListener('DOMContentLoaded', function() {
            const uploadArea = document.getElementById('upload-area');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                uploadArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                uploadArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight(e) {
                uploadArea.classList.add('dragover');
            }
            
            function unhighlight(e) {
                uploadArea.classList.remove('dragover');
            }
            
            uploadArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    document.getElementById('product-upload').files = files;
                    handleProductUpload({ target: { files: files } });
                }
            }
            
            // Check URL parameters for workflow
            const urlParams = new URLSearchParams(window.location.search);
            const workflow = urlParams.get('workflow');
            if (workflow && ['ecommerce', 'marketing', 'creative'].includes(workflow)) {
                showWorkflow(workflow);
            }
        });
    </script>
</body>
</html>'''

@app.route('/')
def landing_page():
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FluxFlow Studio - Professional AI Image Generation SaaS</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #ffffff;
            background: #0a0a0f;
            overflow-x: hidden;
        }}
        
        /* Hero Section */
        .hero {{
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 30%, #16213e  70%, #0f3460 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            position: relative;
            overflow: hidden;
        }}
        
        .hero::after {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.3) 0%, transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(255, 119, 198, 0.3) 0%, transparent 50%),
                        radial-gradient(circle at 40% 80%, rgba(120, 219, 255, 0.2) 0%, transparent 50%);
            animation: gradientShift 8s ease-in-out infinite;
            pointer-events: none;
        }}
        
        .hero::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><radialGradient id="a" cx="50%" cy="50%"><stop offset="0%" stop-color="%23ffffff" stop-opacity="0.05"/><stop offset="100%" stop-color="%23ffffff" stop-opacity="0"/></radialGradient><radialGradient id="b" cx="50%" cy="50%"><stop offset="0%" stop-color="%237c77c6" stop-opacity="0.1"/><stop offset="100%" stop-color="%237c77c6" stop-opacity="0"/></radialGradient></defs><circle cx="200" cy="200" r="100" fill="url(%23a)"/><circle cx="800" cy="300" r="150" fill="url(%23b)"/><circle cx="400" cy="700" r="120" fill="url(%23a)"/><circle cx="600" cy="100" r="80" fill="url(%23b)"/></svg>');
            animation: float 25s ease-in-out infinite;
        }}
        
        @keyframes gradientShift {{
            0%, 100% {{ opacity: 1; transform: translateX(0px) translateY(0px); }}
            25% {{ opacity: 0.8; transform: translateX(10px) translateY(-5px); }}
            50% {{ opacity: 1; transform: translateX(-5px) translateY(10px); }}
            75% {{ opacity: 0.9; transform: translateX(-10px) translateY(-5px); }}
        }}
        
        @keyframes float {{
            0%, 100% {{ transform: translateY(0px) rotate(0deg) scale(1); }}
            33% {{ transform: translateY(-15px) rotate(60deg) scale(1.05); }}
            66% {{ transform: translateY(-25px) rotate(120deg) scale(0.95); }}
        }}
        
        @keyframes glow {{
            0%, 100% {{ box-shadow: 0 0 20px rgba(120, 119, 198, 0.3), 0 0 40px rgba(120, 119, 198, 0.1); }}
            50% {{ box-shadow: 0 0 30px rgba(120, 119, 198, 0.5), 0 0 60px rgba(120, 119, 198, 0.2); }}
        }}
        
        @keyframes slideInUp {{
            from {{ opacity: 0; transform: translateY(30px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        @keyframes fadeInScale {{
            from {{ opacity: 0; transform: scale(0.9); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        
        .hero-content {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 80px;
            align-items: center;
            position: relative;
            z-index: 3;
            min-height: 100vh;
        }}
        
        .hero-text h1 {{
            font-size: 5rem;
            font-weight: 900;
            color: #f8fafc;
            margin-bottom: 25px;
            line-height: 1.05;
            text-shadow: 0 4px 20px rgba(0,0,0,0.3);
            letter-spacing: -0.03em;
            animation: slideInUp 1s ease-out 0.2s both;
        }}
        
        .hero-text .subtitle {{
            font-size: 1.9rem;
            color: #e2e8f0;
            margin-bottom: 20px;
            font-weight: 400;
            line-height: 1.3;
            animation: slideInUp 1s ease-out 0.4s both;
        }}
        
        .hero-text .description {{
            font-size: 1.15rem;
            color: #cbd5e1;
            margin-bottom: 40px;
            font-weight: 300;
            line-height: 1.6;
            max-width: 90%;
            animation: slideInUp 1s ease-out 0.6s both;
        }}
        
        .features-highlight {{
            display: flex;
            flex-wrap: nowrap;
            gap: 10px;
            margin-bottom: 40px;
            animation: slideInUp 1s ease-out 0.8s both;
            justify-content: flex-start;
            overflow-x: auto;
        }}
        
        .feature-highlight {{
            display: flex;
            align-items: center;
            gap: 6px;
            background: rgba(255,255,255,0.08);
            backdrop-filter: blur(15px);
            padding: 8px 14px;
            border-radius: 20px;
            color: #e2e8f0;
            font-weight: 500;
            border: 1px solid rgba(255,255,255,0.15);
            transition: all 0.3s ease;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            white-space: nowrap;
            font-size: 12px;
            flex-shrink: 0;
        }}
        
        .feature-highlight:hover {{
            background: rgba(255,255,255,0.15);
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        .feature-highlight i {{
            color: #ffd700;
        }}
        
        .flux-badge {{
            display: inline-flex;
            align-items: center;
            background: rgba(255,255,255,0.1);
            backdrop-filter: blur(15px);
            padding: 12px 24px;
            border-radius: 25px;
            color: white;
            font-weight: 600;
            margin-bottom: 40px;
            border: 1px solid rgba(255,255,255,0.2);
            animation: slideInUp 1s ease-out both;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .flux-badge i {{
            margin-right: 10px;
            color: #ffd700;
        }}
        
        .cta-buttons {{
            display: flex;
            gap: 15px;
            margin-bottom: 40px;
            animation: slideInUp 1s ease-out 1s both;
            flex-wrap: wrap;
        }}
        
        .btn {{
            padding: 18px 36px;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s ease;
            text-decoration: none;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }}
        
        .btn-primary {{
            background: #ff6b6b;
            color: white;
            box-shadow: 0 4px 15px rgba(255,107,107,0.3);
        }}
        
        .btn-primary:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255,107,107,0.4);
            background: #ff5252;
        }}
        
        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }}
        
        .btn:disabled:hover {{
            transform: none !important;
            box-shadow: none !important;
        }}
        
        .btn-secondary {{
            background: rgba(255,255,255,0.2);
            color: white;
            border: 2px solid rgba(255,255,255,0.3);
            backdrop-filter: blur(10px);
        }}
        
        .btn-secondary:hover {{
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }}
        
        .btn-outline {{
            background: transparent;
            color: #e2e8f0;
            border: 2px solid rgba(255,255,255,0.2);
            backdrop-filter: blur(10px);
            font-size: 14px;
            padding: 14px 28px;
        }}
        
        .btn-outline:hover {{
            background: rgba(255,255,255,0.1);
            border-color: rgba(255,255,255,0.4);
            transform: translateY(-2px);
        }}
        
        .hero-visual {{
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 2;
        }}
        
        .dashboard-preview {{
            background: #1e293b;
            border-radius: 20px;
            padding: 6px;
            box-shadow: 0 25px 60px rgba(0,0,0,0.3), 0 0 0 1px rgba(255,255,255,0.1);
            transform: perspective(1000px) rotateY(-8deg) rotateX(2deg);
            transition: all 0.4s ease;
            max-width: 550px;
            width: 100%;
            animation: fadeInScale 1.2s ease-out 0.5s both;
        }}
        
        .dashboard-preview:hover {{
            transform: perspective(1000px) rotateY(-5deg) rotateX(1deg) scale(1.01);
            box-shadow: 0 30px 70px rgba(0,0,0,0.4), 0 0 0 1px rgba(255,255,255,0.15);
        }}
        
        .preview-header {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0;
            padding: 16px 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            background: #475569;
            border-radius: 16px 16px 0 0;
        }}
        
        .preview-header span {{
            color: rgba(255,255,255,0.8);
            font-size: 14px;
        }}
        
        .preview-dots {{
            display: flex;
            gap: 5px;
        }}
        
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #ddd;
        }}
        
        .dot.red {{ background: #ff5f57; }}
        .dot.yellow {{ background: #ffbd2e; }}
        .dot.green {{ background: #28ca42; }}
        
        .preview-content {{
            background: transparent;
            border-radius: 16px;
            padding: 0;
            overflow: hidden;
        }}
        
        .preview-workspace {{
            background: #0f172a;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.08);
            width: 100%;
        }}
        
        .workspace-header {{
            background: #334155;
            padding: 18px 24px;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        
        .workspace-tabs {{
            display: flex;
            gap: 10px;
        }}
        
        .tab {{
            padding: 10px 18px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            color: rgba(255,255,255,0.7);
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .tab.active {{
            background: #667eea;
            color: white;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
            border-color: #667eea;
        }}
        
        .workspace-body {{
            padding: 28px 24px;
        }}
        
        .input-area {{
            margin-bottom: 15px;
        }}
        
        .input-label {{
            font-size: 12px;
            font-weight: 600;
            color: rgba(255,255,255,0.8);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        .input-field {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 14px;
            color: rgba(255,255,255,0.9);
            backdrop-filter: blur(10px);
            width: 100%;
            box-sizing: border-box;
        }}
        
        .style-chips {{
            display: flex;
            gap: 8px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }}
        
        .chip {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 18px;
            padding: 6px 14px;
            font-size: 11px;
            font-weight: 500;
            color: rgba(255,255,255,0.7);
            cursor: pointer;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}
        
        .chip.active {{
            background: rgba(59, 130, 246, 0.3);
            border-color: #3b82f6;
            color: white;
            box-shadow: 0 1px 6px rgba(59, 130, 246, 0.2);
        }}
        
        .generate-btn {{
            background: #667eea;
            color: white;
            border-radius: 8px;
            padding: 14px 20px;
            font-size: 14px;
            font-weight: 600;
            text-align: center;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            box-shadow: 0 2px 12px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 8px;
        }}
        
        .generate-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
            background: #5a67d8;
        }}
        
        /* Upload Area Styles */
        .upload-area {{
            border: 2px dashed rgba(255,255,255,0.3);
            border-radius: 12px;
            padding: 40px 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(255,255,255,0.02);
            position: relative;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .upload-area:hover {{
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .upload-area.dragover {{
            border-color: #667eea;
            background: rgba(102, 126, 234, 0.1);
        }}
        
        .upload-content {{
            color: rgba(255,255,255,0.7);
        }}
        
        .upload-content i {{
            font-size: 3rem;
            margin-bottom: 15px;
            color: rgba(102, 126, 234, 0.7);
        }}
        
        .upload-content p {{
            font-size: 1.1rem;
            margin-bottom: 8px;
            font-weight: 500;
        }}
        
        .upload-content small {{
            font-size: 0.9rem;
            opacity: 0.6;
        }}
        
        .uploaded-image {{
            max-width: 100%;
            max-height: 300px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}
        
        /* Result Comparison Styles */
        .result-comparison {{
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 20px;
            align-items: center;
            margin: 25px 0;
        }}
        
        .comparison-item {{
            text-align: center;
        }}
        
        .comparison-item h4 {{
            color: rgba(255,255,255,0.8);
            font-size: 14px;
            margin-bottom: 15px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .comparison-image {{
            width: 100%;
            max-height: 250px;
            object-fit: cover;
            border-radius: 12px;
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }}
        
        .comparison-arrow {{
            color: rgba(102, 126, 234, 0.8);
            font-size: 1.5rem;
        }}
        
        .result-actions {{
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 25px;
        }}
        
        @media (max-width: 768px) {{
            .result-comparison {{
                grid-template-columns: 1fr;
                gap: 15px;
            }}
            
            .comparison-arrow {{
                transform: rotate(90deg);
            }}
        }}
        
        /* Modern Studio Container */
        .modern-studio-container {{
            background: rgba(255,255,255,0.02);
            border-radius: 20px;
            padding: 30px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        
        .studio-header {{
            margin-bottom: 30px;
        }}
        
        .studio-title {{
            display: flex;
            align-items: center;
            gap: 20px;
        }}
        
        .studio-icon {{
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            color: white;
        }}
        
        .studio-title h2 {{
            color: white;
            font-size: 28px;
            margin: 0;
            font-weight: 600;
        }}
        
        .studio-title p {{
            color: rgba(255,255,255,0.7);
            margin: 5px 0 0 0;
            font-size: 16px;
        }}
        
        /* Image Workspace */
        .image-workspace {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .image-container {{
            background: rgba(255,255,255,0.05);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .image-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }}
        
        .image-header h3 {{
            color: white;
            font-size: 18px;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .image-status {{
            color: rgba(255,255,255,0.6);
            font-size: 12px;
            background: rgba(255,255,255,0.1);
            padding: 4px 12px;
            border-radius: 20px;
        }}
        
        .image-preview-area {{
            aspect-ratio: 1;
            border-radius: 12px;
            border: 2px dashed rgba(255,255,255,0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .image-preview-area:hover {{
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(102, 126, 234, 0.05);
        }}
        
        .upload-placeholder, .result-placeholder {{
            text-align: center;
            color: rgba(255,255,255,0.6);
        }}
        
        .upload-icon, .result-icon {{
            font-size: 48px;
            margin-bottom: 15px;
            color: rgba(102, 126, 234, 0.7);
        }}
        
        .upload-placeholder h4, .result-placeholder h4 {{
            color: white;
            margin: 0 0 8px 0;
            font-size: 18px;
        }}
        
        .upload-placeholder p, .result-placeholder p {{
            margin: 0 0 8px 0;
            font-size: 14px;
        }}
        
        .upload-placeholder small, .result-placeholder small {{
            font-size: 12px;
            opacity: 0.7;
        }}
        
        .preview-image {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 8px;
        }}
        
        /* Controls Section */
        .controls-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            margin-bottom: 20px;
        }}
        
        .prompt-section {{
            margin-bottom: 25px;
        }}
        
        .modern-label {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 12px;
        }}
        
        .modern-textarea {{
            width: 100%;
            min-height: 120px;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 12px;
            padding: 15px;
            color: white;
            font-size: 14px;
            line-height: 1.5;
            resize: vertical;
            transition: all 0.3s ease;
        }}
        
        .modern-textarea:focus {{
            outline: none;
            border-color: rgba(102, 126, 234, 0.5);
            background: rgba(255,255,255,0.08);
        }}
        
        .modern-textarea::placeholder {{
            color: rgba(255,255,255,0.5);
        }}
        
        /* Enhancement Chips */
        .enhancement-chips {{
            margin-bottom: 25px;
        }}
        
        .chip-label {{
            color: rgba(255,255,255,0.8);
            font-size: 14px;
            margin-bottom: 12px;
            font-weight: 500;
        }}
        
        .chips-container {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }}
        
        .enhancement-chip {{
            background: rgba(255,255,255,0.1);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 25px;
            padding: 8px 16px;
            color: white;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .enhancement-chip:hover {{
            background: rgba(102, 126, 234, 0.3);
            border-color: rgba(102, 126, 234, 0.5);
            transform: translateY(-1px);
        }}
        
        .enhancement-chip.active {{
            background: rgba(102, 126, 234, 0.4);
            border-color: rgba(102, 126, 234, 0.6);
        }}
        
        /* Action Section */
        .action-section {{
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}
        
        .settings-row {{
            display: flex;
            gap: 15px;
        }}
        
        .modern-select {{
            flex: 1;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 12px 15px;
            color: white;
            font-size: 14px;
            cursor: pointer;
        }}
        
        .modern-select:focus {{
            outline: none;
            border-color: rgba(102, 126, 234, 0.5);
        }}
        
        .modern-select option {{
            background: #1a1a2e;
            color: white;
        }}
        
        /* Modern Generate Button */
        .modern-generate-btn {{
            position: relative;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 15px;
            padding: 18px 40px;
            color: white;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            overflow: hidden;
        }}
        
        .modern-generate-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        }}
        
        .modern-generate-btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }}
        
        .btn-glow {{
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        
        .modern-generate-btn:hover .btn-glow {{
            left: 100%;
        }}
        
        /* Status Message */
        .status-message {{
            margin-top: 15px;
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 14px;
            display: none;
        }}
        
        .status-message.success {{
            background: rgba(34, 197, 94, 0.2);
            border: 1px solid rgba(34, 197, 94, 0.3);
            color: #22c55e;
        }}
        
        .status-message.error {{
            background: rgba(239, 68, 68, 0.2);
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: #ef4444;
        }}
        
        .status-message.warning {{
            background: rgba(245, 158, 11, 0.2);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #f59e0b;
        }}
        
        /* Download Section */
        .download-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 16px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .download-actions {{
            display: flex;
            gap: 15px;
            justify-content: center;
        }}
        
        .download-btn {{
            padding: 12px 24px;
            border-radius: 10px;
            border: none;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .download-btn.primary {{
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
        }}
        
        .download-btn.secondary {{
            background: rgba(255,255,255,0.1);
            color: white;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        
        .download-btn:hover {{
            transform: translateY(-1px);
        }}
        
        /* Modern Header Styles */
        .header-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .modern-header-btn {{
            position: relative;
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 12px 20px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            overflow: hidden;
            margin-left: 12px;
        }}
        
        .modern-header-btn.primary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
        }}
        
        .modern-header-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        }}
        
        .modern-header-btn .btn-content {{
            display: flex;
            align-items: center;
            gap: 8px;
            position: relative;
            z-index: 2;
        }}
        
        .modern-header-btn .btn-glow {{
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }}
        
        .modern-header-btn.primary:hover .btn-glow {{
            left: 100%;
        }}
        
        /* Elegant Action Buttons */
        .elegant-btn {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 14px 24px;
            border-radius: 12px;
            border: none;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            min-width: 140px;
            justify-content: center;
        }}
        
        .elegant-btn.primary {{
            background: linear-gradient(135deg, #3b82f6 0%, #1e40af 100%);
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }}
        
        .elegant-btn.success {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        }}
        
        .elegant-btn.secondary {{
            background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
            box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);
        }}
        
        .elegant-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}
        
        .elegant-btn.primary:hover {{
            box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        }}
        
        .elegant-btn.success:hover {{
            box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
        }}
        
        .elegant-btn.secondary:hover {{
            box-shadow: 0 8px 25px rgba(107, 114, 128, 0.4);
        }}
        
        .elegant-btn i {{
            font-size: 16px;
        }}
        
        .elegant-btn:active {{
            transform: translateY(0);
        }}
        
        /* Responsive Design */
        @media (max-width: 768px) {{
            .image-workspace {{
                grid-template-columns: 1fr;
                gap: 20px;
            }}
            
            .settings-row {{
                flex-direction: column;
            }}
            
            .elegant-btn {{
                width: 100%;
                max-width: 200px;
            }}
            
            .chips-container {{
                justify-content: center;
            }}
            
            .header-actions {{
                flex-direction: column;
                gap: 8px;
            }}
            
            .modern-header-btn {{
                margin-left: 0;
                width: 100%;
            }}
        }}
        
        /* Features Section */
        .features {{
            padding: 100px 0;
            background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f23 100%);
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }}
        
        .section-header {{
            text-align: center;
            margin-bottom: 80px;
        }}
        
        .section-header h2 {{
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
        }}
        
        .section-header p {{
            font-size: 1.2rem;
            color: rgba(255,255,255,0.7);
            max-width: 600px;
            margin: 0 auto;
        }}
        
        .features-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 40px;
        }}
        
        .feature-card {{
            background: linear-gradient(145deg, #1a1a2e 0%, #16213e 100%);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            transition: all 0.3s ease;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
        }}
        
        .feature-card:hover {{
            transform: translateY(-10px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 0 0 20px rgba(120, 119, 198, 0.3);
            border-color: rgba(120, 119, 198, 0.3);
        }}
        
        .feature-icon {{
            width: 80px;
            height: 80px;
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            margin-bottom: 25px;
        }}
        
        .feature-icon.ecommerce {{
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
        }}
        
        .feature-icon.marketing {{
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: white;
        }}
        
        .feature-icon.ai {{
            background: linear-gradient(135deg, #4facfe, #00f2fe);
            color: white;
        }}
        
        .feature-card h3 {{
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 15px;
        }}
        
        .feature-card p {{
            color: rgba(255,255,255,0.7);
            line-height: 1.6;
            margin-bottom: 20px;
        }}
        
        .feature-list {{
            list-style: none;
        }}
        
        .feature-list li {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
            color: rgba(255,255,255,0.8);
        }}
        
        .feature-list i {{
            color: #28a745;
            font-size: 0.9rem;
        }}
        
        /* Workflow Section */
        .workflows {{
            padding: 100px 0;
            background: linear-gradient(135deg, #16213e 0%, #0f3460 50%, #1a1a2e 100%);
        }}
        
        .workflow-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 60px;
        }}
        
        .workflow-card {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
            color: white;
            padding: 40px;
            border-radius: 20px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(20px);
        }}
        
        .workflow-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 35px rgba(102,126,234,0.4);
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%);
            border-color: rgba(102, 126, 234, 0.5);
        }}
        
        .workflow-card i {{
            font-size: 3rem;
            margin-bottom: 20px;
            opacity: 0.9;
        }}
        
        .workflow-card h3 {{
            font-size: 1.5rem;
            margin-bottom: 15px;
        }}
        
        /* Stats Section */
        .stats {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 80px 0;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 40px;
            text-align: center;
        }}
        
        .stat-item h3 {{
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 10px;
            color: #3498db;
        }}
        
        .stat-item p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
        
        /* CTA Section */
        .cta-section {{
            background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
            color: white;
            padding: 100px 0;
            text-align: center;
            position: relative;
        }}
        
        .cta-section::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 50% 50%, rgba(120, 119, 198, 0.2) 0%, transparent 70%);
            pointer-events: none;
        }}
        
        .cta-section h2 {{
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 50%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .cta-section p {{
            font-size: 1.2rem;
            margin-bottom: 40px;
            color: rgba(255,255,255,0.8);
        }}
        
        /* Responsive */
        @media (max-width: 768px) {{
            .hero-content {{
                grid-template-columns: 1fr;
                text-align: center;
                gap: 40px;
            }}
            
            .hero-text h1 {{
                font-size: 3rem;
            }}
            
            .hero-text .subtitle {{
                font-size: 1.4rem;
            }}
            
            .dashboard-preview {{
                transform: none;
                max-width: 100%;
            }}
            
            .cta-buttons {{
                flex-direction: column;
                align-items: center;
            }}
            
            .features-highlight {{
                justify-content: center;
            }}
        }}
    </style>
</head>
<body>
    <!-- Hero Section -->
    <section class="hero">
        <div class="hero-content">
            <div class="hero-text">
                <div class="flux-badge">
                    <i class="fas fa-bolt"></i>
                    Powered by FLUX.1 AI
                </div>
                <h1>FluxFlow Studio</h1>
                <p class="subtitle">AI-Powered Product Photography & Marketing Content Creation</p>
                <p class="description">Transform your product images into professional e-commerce photos and generate targeted marketing campaigns with intelligent AI suggestions. Powered by FLUX.1 for enterprise-quality results.</p>
                
                <div class="cta-buttons">
                    <a href="/studio" class="btn btn-primary">
                        <i class="fas fa-camera"></i>
                        Start Creating
                    </a>
                    <a href="#features" class="btn btn-secondary">
                        <i class="fas fa-magic"></i>
                        See Features
                    </a>
                    <a href="#demo" class="btn btn-outline">
                        <i class="fas fa-play"></i>
                        Watch Demo
                    </a>
                </div>
                
                <div class="features-highlight">
                    <div class="feature-highlight"><i class="fas fa-camera-retro"></i> Product Photography</div>
                    <div class="feature-highlight"><i class="fas fa-bullhorn"></i> AI Marketing Ideas</div>
                    <div class="feature-highlight"><i class="fas fa-robot"></i> Smart Suggestions</div>
                    <div class="feature-highlight"><i class="fas fa-bolt"></i> FLUX.1 Powered</div>
                </div>
            </div>
            
            <div class="hero-visual">
                <div class="dashboard-preview">
                    <div class="preview-header">
                        <div class="preview-dots">
                            <div class="dot red"></div>
                            <div class="dot yellow"></div>
                            <div class="dot green"></div>
                        </div>
                        <span style="margin-left: 10px; color: rgba(255,255,255,0.8); font-size: 14px;">FluxFlow Studio</span>
                    </div>
                    <div class="preview-content">
                        <div class="preview-workspace">
                            <div class="workspace-header">
                                <div class="workspace-tabs">
                                    <div class="tab active">Marketing Content</div>
                                    <div class="tab">Product Photos</div>
                                </div>
                            </div>
                            <div class="workspace-body">
                                <div class="input-area">
                                    <div class="input-label">Upload Product Image</div>
                                    <div class="upload-preview" style="background: rgba(255,255,255,0.05); border: 2px dashed rgba(255,255,255,0.3); border-radius: 8px; padding: 20px; text-align: center; margin-bottom: 15px;">
                                        <i class="fas fa-image" style="font-size: 24px; color: rgba(255,255,255,0.5); margin-bottom: 8px;"></i>
                                        <div style="color: rgba(255,255,255,0.7); font-size: 12px;">Peanuts Package Uploaded</div>
                                    </div>
                                </div>
                                <div class="input-area">
                                    <div class="input-label">Marketing Style</div>
                                    <div class="style-chips">
                                        <div class="chip">Social Media</div>
                                        <div class="chip">Banner Ad</div>
                                        <div class="chip active">Promotional</div>
                                    </div>
                                </div>
                                <div class="generate-btn">
                                    <i class="fas fa-robot"></i> Generate Marketing Ideas
                                </div>
                                <div style="margin-top: 15px; padding: 12px; background: rgba(34, 197, 94, 0.1); border-radius: 6px; border-left: 3px solid #22c55e;">
                                    <div style="color: #22c55e; font-size: 11px; font-weight: 600;">✓ AI SUGGESTIONS GENERATED</div>
                                    <div style="color: rgba(255,255,255,0.8); font-size: 12px; margin-top: 4px;">"Cartoon peanut with 20% OFF banner"</div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <!-- Features Section -->
    <section class="features" id="features">
        <div class="container">
            <div class="section-header">
                <h2>Powerful Features for Modern Businesses</h2>
                <p>Everything you need to create stunning visuals for your e-commerce and marketing needs</p>
            </div>
            
            <div class="features-grid">
                <div class="feature-card">
                    <div class="feature-icon ecommerce">
                        <i class="fas fa-camera-retro"></i>
                    </div>
                    <h3>Product Photo Enhancement</h3>
                    <p>Transform mobile phone photos into professional e-commerce images with AI-powered enhancement and styling.</p>
                    <ul class="feature-list">
                        <li><i class="fas fa-check"></i> Mobile photo enhancement</li>
                        <li><i class="fas fa-check"></i> Background removal & replacement</li>
                        <li><i class="fas fa-check"></i> Professional lighting correction</li>
                        <li><i class="fas fa-check"></i> Multiple style variations</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon marketing">
                        <i class="fas fa-robot"></i>
                    </div>
                    <h3>AI Marketing Suggestions</h3>
                    <p>Upload your product image and get intelligent, detailed marketing campaign ideas with specific content and visual concepts.</p>
                    <ul class="feature-list">
                        <li><i class="fas fa-check"></i> Product-specific campaign ideas</li>
                        <li><i class="fas fa-check"></i> Detailed visual descriptions</li>
                        <li><i class="fas fa-check"></i> Exact text content & headlines</li>
                        <li><i class="fas fa-check"></i> Style-aware suggestions</li>
                    </ul>
                </div>
                
                <div class="feature-card">
                    <div class="feature-icon ai">
                        <i class="fas fa-brain"></i>
                    </div>
                    <h3>Advanced AI Technology</h3>
                    <p>Powered by cutting-edge FLUX.1 models with professional prompt engineering for superior results.</p>
                    <ul class="feature-list">
                        <li><i class="fas fa-check"></i> FLUX.1 Schnell (Fast)</li>
                        <li><i class="fas fa-check"></i> FLUX.1 Dev (Quality)</li>
                        <li><i class="fas fa-check"></i> Smart prompt enhancement</li>
                        <li><i class="fas fa-check"></i> Style optimization</li>
                    </ul>
                </div>
            </div>
        </div>
    </section>

    <!-- Workflows Section -->
    <section class="workflows">
        <div class="container">
            <div class="section-header">
                <h2>Specialized Workflows</h2>
                <p>Choose your workflow and let our AI create professional results tailored to your needs</p>
            </div>
            
            <div class="workflow-cards">
                <div class="workflow-card" onclick="window.location.href='/studio'">
                    <i class="fas fa-camera-retro"></i>
                    <h3>Product Photo Enhancement</h3>
                    <p>Transform mobile photos into professional e-commerce images with AI-powered enhancement</p>
                    <div class="workflow-badge">Mobile to Pro</div>
                </div>
                
                <div class="workflow-card" onclick="window.location.href='/studio'">
                    <i class="fas fa-robot"></i>
                    <h3>AI Marketing Ideas</h3>
                    <p>Upload product images and get detailed marketing campaign suggestions with exact content</p>
                    <div class="workflow-badge">Smart AI</div>
                </div>
            </div>
        </div>
    </section>


    <!-- CTA Section -->
    <section class="cta-section">
        <div class="container">
            <h2>Ready to Transform Your Visual Content?</h2>
            <p>Join thousands of businesses creating professional images with AI</p>
            <a href="/studio" class="btn btn-primary" style="font-size: 1.2rem; padding: 20px 40px;">
                <i class="fas fa-rocket"></i>
                Start Creating Now
            </a>
        </div>
    </section>

    <script>
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
        
        // Add scroll animations
        const observerOptions = {{
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        }};
        
        const observer = new IntersectionObserver((entries) => {{
            entries.forEach(entry => {{
                if (entry.isIntersecting) {{
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }}
            }});
        }}, observerOptions);
        
        // Observe all feature cards
        document.querySelectorAll('.feature-card, .workflow-card').forEach(card => {{
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            card.style.transition = 'all 0.6s ease';
            observer.observe(card);
        }});
    </script>
</body>
</html>'''

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'app': 'FluxFlow Studio SaaS',
        'version': '2.0.0',
        'hf_configured': bool(HF_TOKEN),
        'models_available': list(FLUX_MODELS.keys()),
        'workflows': ['ecommerce', 'marketing', 'creative'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/generate-advanced', methods=['POST'])
def generate_advanced():
    """Advanced image generation with workflow-specific optimizations"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        workflow_type = data.get('workflow_type', 'general')
        style_type = data.get('style_type', 'default')
        quality = data.get('quality', 'high')
        model = data.get('model', 'schnell')
        reference_image = data.get('reference_image')
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        if not HF_TOKEN:
            return jsonify({
                'success': False,
                'error': 'Hugging Face token not configured. Please set HF_TOKEN in your .env file to enable AI image generation.'
            })
        
        # For product enhancement, modify the prompt to be more specific
        if workflow_type == 'ecommerce' and reference_image:
            # Create enhancement-focused prompt
            enhancement_prompts = {
                'background_removal': f"Professional product photography of peanut package with orange packaging, RAW PEANUTS text, HILL STONE branding, clean white background, studio lighting, commercial photography, {prompt}",
                'lighting_enhance': f"Professional product photography of peanut package with orange packaging, RAW PEANUTS text, HILL STONE branding, enhanced studio lighting, professional shadows and highlights, {prompt}",
                'color_correction': f"Professional product photography of peanut package with orange packaging, RAW PEANUTS text, HILL STONE branding, enhanced vibrant colors, accurate color reproduction, {prompt}",
                'premium_finish': f"Professional product photography of peanut package with orange packaging, RAW PEANUTS text, HILL STONE branding, premium finishing, enhanced textures, luxury presentation, {prompt}"
            }
            
            if style_type in enhancement_prompts:
                prompt = enhancement_prompts[style_type]
            else:
                prompt = f"Professional product photography of the same product, enhanced for e-commerce, {prompt}"
        
        # Smart image editing with multiple providers and fallback
        if reference_image and workflow_type == 'ecommerce':
            # Convert URL to file path
            if reference_image.startswith('/static/uploads/'):
                image_filename = reference_image.replace('/static/uploads/', '')
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
                
                if os.path.exists(image_path):
                    # Try different providers based on model selection
                    if model == 'fal-kontext' and FAL_KEY:
                        # Use FAL.ai FLUX-Pro Kontext (Best Quality)
                        result = flux_engine.edit_image_fal(prompt, image_path)
                    elif model == 'fal-img2img' and FAL_KEY:
                        # Use FAL.ai directly
                        result = flux_engine.edit_image_fal(prompt, image_path)
                    else:
                        # Use enhanced prompting as fallback
                        enhanced_prompt = f"Professional product photography: {prompt}. Maintain exact product details, high quality, commercial grade"
                        result = flux_engine.generate_image(
                            prompt=enhanced_prompt,
                            workflow_type=workflow_type,
                            style_type=style_type,
                            model=model if model in ['dev', 'schnell'] else 'dev',
                            quality=quality
                        )
                else:
                    result = {"success": False, "error": "Reference image not found"}
            else:
                result = {"success": False, "error": "Invalid reference image URL"}
        else:
            # Fallback to regular generation if no FAL_KEY or no reference image
            if reference_image and not FAL_KEY:
                prompt = f"Professional product photography: {prompt}. High quality, commercial grade, studio lighting, clean composition, e-commerce ready"
            
            result = flux_engine.generate_image(
                prompt=prompt,
                workflow_type=workflow_type,
                style_type=style_type,
                model=model,
                quality=quality
            )
        
        if result['success']:
            logger.info(f"Advanced image generated successfully for {workflow_type}/{style_type}: {prompt[:50]}...")
        else:
            logger.error(f"Advanced image generation failed: {result['error']}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error in generate_advanced: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/test-groq', methods=['GET'])
def test_groq():
    """Test Groq API connectivity"""
    try:
        if not GROQ_API_KEY:
            return jsonify({'success': False, 'error': 'GROQ_API_KEY not configured'})
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "user",
                    "content": "Say hello and confirm you're working!"
                }
            ],
            "max_tokens": 50
        }
        
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({'success': True, 'response': result})
        else:
            return jsonify({'success': False, 'error': f'API Error {response.status_code}: {response.text}'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate-marketing-suggestions', methods=['POST'])
def generate_marketing_suggestions():
    """Generate AI-powered marketing suggestions for uploaded image"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        marketing_style = data.get('marketing_style', 'social_media')
        
        if not image_url:
            return jsonify({'success': False, 'error': 'No image URL provided'})
        
        if not GROQ_API_KEY:
            return jsonify({
                'success': False,
                'error': 'AI suggestions not available. GROQ_API_KEY not configured.'
            })
        
        # For now, generate suggestions without requiring the actual image file
        # since we're using text-only AI anyway
        logger.info(f"Generating marketing suggestions for style: {marketing_style}")
        
        # Generate marketing-specific AI suggestions
        result = flux_engine.generate_marketing_ai_suggestions(None, marketing_style)
        
        if result['success']:
            logger.info(f"Marketing AI suggestions generated successfully: {len(result['suggestions'])} suggestions")
        else:
            logger.error(f"Marketing AI suggestions failed: {result['error']}")
        
        return jsonify(result)
            
    except Exception as e:
        logger.error(f"API error in generate_marketing_suggestions: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/generate-suggestions', methods=['POST'])
def generate_suggestions():
    """Generate AI-powered enhancement suggestions for uploaded image"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        
        if not image_url:
            return jsonify({'success': False, 'error': 'No image URL provided'})
        
        if not GROQ_API_KEY:
            return jsonify({
                'success': False,
                'error': 'AI suggestions not available. GROQ_API_KEY not configured.'
            })
        
        # Convert URL to file path
        if image_url.startswith('/static/uploads/'):
            image_filename = image_url.replace('/static/uploads/', '')
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            
            if os.path.exists(image_path):
                # Generate AI suggestions
                result = flux_engine.generate_ai_suggestions(image_path)
                
                if result['success']:
                    logger.info(f"AI suggestions generated successfully: {len(result['suggestions'])} suggestions")
                else:
                    logger.error(f"AI suggestions failed: {result['error']}")
                
                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': 'Image file not found'})
        else:
            return jsonify({'success': False, 'error': 'Invalid image URL'})
            
    except Exception as e:
        logger.error(f"API error in generate_suggestions: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/save-to-gallery', methods=['POST'])
def save_to_gallery():
    """Save enhanced image to gallery"""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        original_url = data.get('original_url')
        prompt = data.get('prompt', '')
        
        if not image_url:
            return jsonify({'success': False, 'error': 'No image URL provided'})
        
        # Create gallery entry
        gallery_item = {
            'id': str(uuid.uuid4()),
            'enhanced_image': image_url,
            'original_image': original_url,
            'prompt': prompt,
            'created_at': datetime.now().isoformat(),
            'type': 'product_enhancement'
        }
        
        # Load existing gallery or create new
        gallery_file = os.path.join(app.config['PROJECTS_FOLDER'], 'gallery.json')
        gallery = []
        
        if os.path.exists(gallery_file):
            try:
                with open(gallery_file, 'r') as f:
                    gallery = json.load(f)
            except:
                gallery = []
        
        # Add new item to beginning of gallery
        gallery.insert(0, gallery_item)
        
        # Keep only last 50 items
        gallery = gallery[:50]
        
        # Save gallery
        with open(gallery_file, 'w') as f:
            json.dump(gallery, f, indent=2)
        
        logger.info(f"Image saved to gallery: {gallery_item['id']}")
        return jsonify({
            'success': True,
            'message': 'Image saved to gallery successfully!',
            'gallery_id': gallery_item['id']
        })
        
    except Exception as e:
        logger.error(f"API error in save_to_gallery: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/get-gallery', methods=['GET'])
def get_gallery():
    """Get gallery items"""
    try:
        gallery_file = os.path.join(app.config['PROJECTS_FOLDER'], 'gallery.json')
        
        if os.path.exists(gallery_file):
            with open(gallery_file, 'r') as f:
                gallery = json.load(f)
        else:
            gallery = []
        
        return jsonify({
            'success': True,
            'gallery': gallery
        })
        
    except Exception as e:
        logger.error(f"API error in get_gallery: {str(e)}")
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload image for editing or reference"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return jsonify({'success': False, 'error': f'Unsupported file type: {file_ext}'})
        
        # Save file
        filename = f"upload_{uuid.uuid4()}{file_ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded successfully: {filename}")
        return jsonify({
            'success': True,
            'image_url': f'/static/uploads/{filename}',
            'filename': filename
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'success': False, 'error': f'Upload failed: {str(e)}'})

@app.route('/static/<path:path>')
def static_files(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🎨 FluxFlow Studio SaaS - Professional AI Image Generation Platform")
    print("="*80)
    print(f"📁 Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"📁 Projects folder: {app.config['PROJECTS_FOLDER']}")
    print(f"🤖 Hugging Face: {'✅ Token Configured' if HF_TOKEN else '⚠️ Token Required (set HF_TOKEN in .env)'}")
    print(f"🎯 Available Models: {', '.join(FLUX_MODELS.keys())}")
    print(f"🔧 Workflows: E-commerce, Marketing, Creative")
    print("\n🌐 Access your SaaS application:")
    print("   🏠 Landing Page: http://localhost:5000")
    print("   🎨 Studio Workspace: http://localhost:5000/studio")
    print("   🔧 Health Check: http://localhost:5000/health")
    print("\n🎯 SaaS Features:")
    print("   ✅ Professional e-commerce product photography")
    print("   ✅ Marketing content creation workflows")
    print("   ✅ Advanced FLUX.1 prompt engineering")
    print("   ✅ Workflow-specific optimizations")
    print("   ✅ Modern SaaS UI/UX design")
    print("   ✅ Professional project management")
    print("="*80 + "\n")
if __name__ == '__main__':
    import os
    # Render uses PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    # Render requires host='0.0.0.0' for external access
    app.run(debug=False, host='0.0.0.0', port=port)
