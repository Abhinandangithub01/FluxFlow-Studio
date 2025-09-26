// FluxFlow Studio Pro - Professional App Interface (Fixed Version)
// Enhanced with left navigation and feature-based workflow

class FluxFlowApp {
    constructor() {
        this.currentProject = null;
        this.currentFeature = 'professional-design';
        this.currentIteration = null;
        this.isProcessing = false;
        
        this.initializeApp();
    }

    initializeApp() {
        this.setupEventListeners();
        this.loadSuggestions();
        this.checkHealth();
        console.log('🚀 FluxFlow Studio Pro initialized!');
    }

    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            const data = await response.json();
            
            if (data.success) {
                console.log('✅ Backend health check passed');
                console.log(`🤖 HF FLUX available: ${data.hf_flux_available}`);
            }
        } catch (error) {
            console.warn('⚠️ Backend health check failed:', error);
        }
    }

    setupEventListeners() {
        // File upload handlers for all upload inputs
        const uploadInputs = [
            'designUpload',
            'cinematicUpload', 
            'marketingUpload',
            'batchUpload'
        ];
        
        uploadInputs.forEach(inputId => {
            const input = document.getElementById(inputId);
            if (input) {
                input.addEventListener('change', (e) => this.handleImageUpload(e));
            }
        });

        // Drag and drop
        const uploadAreas = document.querySelectorAll('.upload-area');
        uploadAreas.forEach(area => {
            area.addEventListener('dragover', (e) => {
                e.preventDefault();
                area.style.borderColor = '#4f46e5';
                area.style.background = 'rgba(79, 70, 229, 0.05)';
            });

            area.addEventListener('dragleave', (e) => {
                e.preventDefault();
                area.style.borderColor = '#d1d5db';
                area.style.background = 'transparent';
            });

            area.addEventListener('drop', (e) => {
                e.preventDefault();
                area.style.borderColor = '#d1d5db';
                area.style.background = 'transparent';
                
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.processImageFile(files[0]);
                }
            });
        });

        // Enter key for edit instructions
        const editInstructions = document.getElementById('editInstructions');
        if (editInstructions) {
            editInstructions.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && e.ctrlKey) {
                    this.applyEdit();
                }
            });
        }

        // Apply edit button
        const applyEditBtn = document.getElementById('applyEdit');
        if (applyEditBtn) {
            applyEditBtn.addEventListener('click', () => this.applyEdit());
        }

        // Generate from text button
        const generateTextBtn = document.getElementById('generateFromText');
        if (generateTextBtn) {
            generateTextBtn.addEventListener('click', () => this.generateFromText());
        }

        // Feature navigation
        const featureButtons = document.querySelectorAll('.feature-btn');
        featureButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const feature = e.target.dataset.feature;
                if (feature) {
                    this.switchFeature(feature);
                }
            });
        });
    }

    switchFeature(feature) {
        this.currentFeature = feature;
        
        // Update UI to reflect current feature
        const featureButtons = document.querySelectorAll('.feature-btn');
        featureButtons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.feature === feature);
        });

        // Update content area
        this.updateFeatureContent();
        
        console.log(`Switched to feature: ${feature}`);
    }

    updateFeatureContent() {
        // This would update the main content area based on the selected feature
        // Implementation depends on your HTML structure
        const contentArea = document.getElementById('mainContent');
        if (contentArea) {
            // Update content based on this.currentFeature
        }
    }

    showAlert(message, type = 'success') {
        const alert = document.getElementById('alert');
        if (alert) {
            alert.textContent = message;
            alert.className = `alert ${type} active`;
            
            setTimeout(() => {
                alert.classList.remove('active');
            }, 5000);
        } else {
            // Fallback to console if no alert element
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    showLoading(show = true) {
        const loading = document.getElementById('loading');
        if (loading) {
            loading.classList.toggle('active', show);
        }
        
        // Disable/enable buttons during processing
        const buttons = document.querySelectorAll('button');
        buttons.forEach(btn => {
            btn.disabled = show;
        });
        
        this.isProcessing = show;
    }

    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (file) {
            await this.processImageFile(file);
        }
    }

    async processImageFile(file) {
        try {
            console.log('DEBUG: Processing file:', file.name, file.size, file.type);
            
            // Validate file
            if (!this.validateFile(file)) {
                return;
            }
            
            this.showLoading(true);
            
            // Create project if none exists
            if (!this.currentProject) {
                console.log('DEBUG: Creating new project...');
                await this.createProject();
            }

            console.log('DEBUG: Current project:', this.currentProject);

            const formData = new FormData();
            formData.append('file', file);
            formData.append('project_id', this.currentProject.id);

            console.log('DEBUG: Uploading file to /api/upload-image');

            const response = await fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentIteration = data.iteration;
                this.displayImage(data.image_url);
                this.updateHistory();
                this.loadSuggestions();
                
                this.showAlert('Image uploaded successfully!');
            } else {
                this.showAlert(data.error || 'Failed to upload image', 'error');
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            this.showAlert('Error uploading image', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    validateFile(file) {
        // Check file size (32MB limit)
        const maxSize = 32 * 1024 * 1024;
        if (file.size > maxSize) {
            this.showAlert('File too large. Maximum size is 32MB.', 'error');
            return false;
        }

        // Check file type
        const allowedTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            this.showAlert('Unsupported file type. Please use JPEG, PNG, WebP, or BMP.', 'error');
            return false;
        }

        return true;
    }

    displayImage(imageUrl) {
        // Try different canvas and image combinations based on current feature
        const canvasSelectors = [
            'designCanvas',
            'cinematicCanvas', 
            'marketingCanvas',
            'batchCanvas'
        ];
        
        const imageSelectors = [
            'designImage',
            'cinematicImage',
            'marketingImage',
            'batchImage'
        ];
        
        let displayed = false;
        
        // Try to find and use the appropriate canvas/image combination
        for (let i = 0; i < canvasSelectors.length; i++) {
            const canvas = document.getElementById(canvasSelectors[i]);
            const image = document.getElementById(imageSelectors[i]);
            
            if (canvas && image) {
                image.src = imageUrl;
                image.onload = () => {
                    canvas.style.display = 'block';
                    console.log('Image displayed successfully in', canvasSelectors[i]);
                };
                image.onerror = () => {
                    console.error('Failed to load image:', imageUrl);
                    this.showAlert('Failed to load image', 'error');
                };
                displayed = true;
                break;
            }
        }
        
        // Fallback: create a simple image display
        if (!displayed) {
            const mainContent = document.querySelector('.main-content');
            if (mainContent) {
                const imgElement = document.createElement('img');
                imgElement.src = imageUrl;
                imgElement.style.maxWidth = '100%';
                imgElement.style.marginTop = '20px';
                imgElement.style.borderRadius = '8px';
                mainContent.appendChild(imgElement);
                console.log('Image displayed using fallback method');
            }
        }
    }

    async createProject() {
        try {
            const response = await fetch('/api/create-project', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    name: `${this.getFeatureName()} Project`,
                    type: this.currentFeature.replace('-', '_')
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentProject = data.project;
                console.log('Project created:', this.currentProject);
            } else {
                throw new Error(data.error || 'Failed to create project');
            }
        } catch (error) {
            console.error('Error creating project:', error);
            this.showAlert('Error creating project', 'error');
            throw error;
        }
    }

    async applyEdit() {
        if (!this.currentProject || !this.currentIteration || this.isProcessing) {
            this.showAlert('Please upload an image first', 'error');
            return;
        }

        const editInstructions = document.getElementById('editInstructions');
        const editStrength = document.getElementById('editStrength');
        
        const instruction = editInstructions?.value?.trim();
        if (!instruction) {
            this.showAlert('Please enter edit instructions', 'error');
            return;
        }

        try {
            this.showLoading(true);
            
            const response = await fetch('/api/edit-image', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    project_id: this.currentProject.id,
                    parent_iteration_id: this.currentIteration.id,
                    edit_instruction: instruction,
                    workflow_type: this.currentFeature.replace('-', '_'),
                    edit_strength: editStrength?.value || 'medium',
                    style_options: this.getStyleOptions()
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentIteration = data.iteration;
                this.displayImage(data.image_url);
                this.updateHistory();
                
                // Clear edit instructions
                if (editInstructions) {
                    editInstructions.value = '';
                }
                
                this.showAlert('Edit applied successfully!');
            } else {
                this.showAlert(data.error || 'Failed to apply edit', 'error');
            }
        } catch (error) {
            console.error('Error applying edit:', error);
            this.showAlert('Error applying edit', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    async generateFromText() {
        const textPrompt = prompt('Enter your text prompt:');
        if (!textPrompt?.trim()) return;

        try {
            this.showLoading(true);
            
            // Create project if none exists
            if (!this.currentProject) {
                await this.createProject();
            }
            
            const response = await fetch('/api/generate-from-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    project_id: this.currentProject.id,
                    text_prompt: textPrompt,
                    workflow_type: this.currentFeature.replace('-', '_'),
                    style_options: this.getStyleOptions()
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.currentIteration = data.iteration;
                this.displayImage(data.image_url);
                this.updateHistory();
                this.loadSuggestions();
                
                this.showAlert('Image generated successfully!');
            } else {
                this.showAlert(data.error || 'Failed to generate image', 'error');
            }
        } catch (error) {
            console.error('Error generating image:', error);
            this.showAlert('Error generating image', 'error');
        } finally {
            this.showLoading(false);
        }
    }

    getStyleOptions() {
        const options = {};
        
        switch (this.currentFeature) {
            case 'professional-design':
                options.design_style = 'professional';
                options.target_audience = 'general';
                options.quality = 'high';
                break;
            case 'cinematic-photo':
                options.cinematic_style = 'modern_drama';
                options.mood = 'dramatic';
                options.quality = 'high';
                break;
            case 'marketing-content':
                options.campaign_type = 'social_media';
                options.target_audience = 'general';
                options.brand_style = 'modern';
                options.quality = 'high';
                break;
            default:
                options.style = 'photorealistic';
                options.quality = 'high';
        }
        
        return options;
    }

    getFeatureName() {
        const names = {
            'professional-design': 'Professional Design',
            'cinematic-photo': 'Cinematic Photo',
            'marketing-content': 'Marketing Content',
            'text-to-image': 'Text to Image'
        };
        
        return names[this.currentFeature] || 'Unknown';
    }

    updateHistory() {
        // Update the history/iterations display
        const historyContainer = document.getElementById('historyContainer');
        if (historyContainer && this.currentProject) {
            // Implementation depends on your HTML structure
            console.log('Updating history for project:', this.currentProject.id);
        }
    }

    async loadSuggestions() {
        // Load edit suggestions based on current feature
        try {
            const response = await fetch(`/api/get-edit-types/${this.currentFeature.replace('-', '_')}`);
            const data = await response.json();
            
            if (data.success) {
                this.displaySuggestions(data.edit_types);
            }
        } catch (error) {
            console.warn('Failed to load suggestions:', error);
        }
    }

    displaySuggestions(editTypes) {
        const suggestionsContainer = document.getElementById('suggestionsContainer');
        if (!suggestionsContainer || !editTypes) return;

        // Clear existing suggestions
        suggestionsContainer.innerHTML = '';

        // Create suggestion buttons for each category
        Object.entries(editTypes).forEach(([category, suggestions]) => {
            const categoryDiv = document.createElement('div');
            categoryDiv.className = 'suggestion-category';
            
            const categoryTitle = document.createElement('h4');
            categoryTitle.textContent = category.replace('_', ' ').toUpperCase();
            categoryDiv.appendChild(categoryTitle);

            const suggestionsDiv = document.createElement('div');
            suggestionsDiv.className = 'suggestions-list';

            suggestions.forEach(suggestion => {
                const btn = document.createElement('button');
                btn.className = 'suggestion-btn';
                btn.textContent = suggestion;
                btn.onclick = () => this.applySuggestion(suggestion);
                suggestionsDiv.appendChild(btn);
            });

            categoryDiv.appendChild(suggestionsDiv);
            suggestionsContainer.appendChild(categoryDiv);
        });
    }

    applySuggestion(suggestion) {
        const editInstructions = document.getElementById('editInstructions');
        if (editInstructions) {
            const currentText = editInstructions.value.trim();
            const newText = currentText ? `${currentText}, ${suggestion}` : suggestion;
            editInstructions.value = newText;
            editInstructions.focus();
        }
    }

    // Utility methods
    async downloadImage() {
        if (!this.currentIteration) {
            this.showAlert('No image to download', 'error');
            return;
        }

        try {
            const imageUrl = `/static/uploads/${this.currentIteration.filename}`;
            const link = document.createElement('a');
            link.href = imageUrl;
            link.download = this.currentIteration.filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showAlert('Image downloaded successfully!');
        } catch (error) {
            console.error('Error downloading image:', error);
            this.showAlert('Error downloading image', 'error');
        }
    }

    async shareImage() {
        if (!this.currentIteration) {
            this.showAlert('No image to share', 'error');
            return;
        }

        try {
            const imageUrl = `${window.location.origin}/static/uploads/${this.currentIteration.filename}`;
            
            if (navigator.share) {
                await navigator.share({
                    title: 'FluxFlow Studio Pro - Generated Image',
                    url: imageUrl
                });
            } else {
                // Fallback: copy to clipboard
                await navigator.clipboard.writeText(imageUrl);
                this.showAlert('Image URL copied to clipboard!');
            }
        } catch (error) {
            console.error('Error sharing image:', error);
            this.showAlert('Error sharing image', 'error');
        }
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.fluxApp = new FluxFlowApp();
});

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = FluxFlowApp;
}
