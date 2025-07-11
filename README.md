# AI Virtual Personal Stylist - Fashion Recommendation System

A comprehensive fashion recommendation system that leverages deep learning and computer vision to suggest personalized fashion items based on user preferences. The system processes fashion product images, extracts visual features using pre-trained neural networks, and recommends similar products using cosine similarity.

## 🌟 Features

- **Visual Feature Extraction**: Uses MobileNetV2 for extracting deep visual features from fashion images
- **Personalized Recommendations**: Suggests items based on gender, category, and usage preferences
- **Similarity-Based Matching**: Employs cosine similarity to find visually similar fashion products
- **Interactive Interface**: User-friendly input system for specifying preferences
- **Data Balancing**: Handles class imbalance through intelligent resampling techniques
- **Batch Processing**: Efficient processing of large image datasets

## 🛠️ Technologies Used

- **Python**: Core programming language
- **TensorFlow/Keras**: Deep learning framework for feature extraction
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib & Seaborn**: Data visualization and analysis
- **scikit-learn**: Machine learning utilities and similarity metrics
- **PIL/Pillow**: Image processing and manipulation
- **tqdm**: Progress tracking for batch operations

## 📁 Dataset

The project uses a fashion dataset from Kaggle containing:
- Product images
- Metadata (category, subcategory, gender, color, usage, season)
- Product descriptions and identifiers

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/esraamhmd/AI-Virtual-Personal-Stylist-Clothing-Recommendations-.git
   cd AI-Virtual-Personal-Stylist-Clothing-Recommendations-
   ```

2. **Install required dependencies**
   ```bash
   pip install tensorflow pandas numpy matplotlib seaborn scikit-learn pillow tqdm
   ```

3. **Download the dataset**
   - Download the fashion dataset from Kaggle
   - Place the CSV file and images in the appropriate directories
   - Update file paths in the notebook accordingly

## 📊 Project Workflow

### 1. Data Analysis & Exploration
- Dataset structure examination (shape, columns, data types)
- Missing value analysis across all columns
- Category and subcategory distribution analysis
- Gender and color distribution visualization
- Class imbalance detection

### 2. Data Cleaning
- **Missing Value Handling**:
  - Removal of rows with missing critical fields
  - Categorical fields filled with 'unknown'
  - Mode values for numerical fields
- **Data Type Conversion**: ID formatting for image file matching
- **Duplicate Removal**: Identification and removal of duplicate entries

### 3. Data Balancing
- **Category Filtering**: Minimum threshold of 100 samples per category
- **Resampling Implementation**:
  - Target size of 1000 samples per category
  - Upsampling for categories with fewer samples
  - Downsampling for categories with excess samples
- **Balance Verification**: Before/after comparison visualizations

### 4. Data Processing & Feature Extraction
- **Image Preprocessing**: Resizing to 224×224, normalization
- **Label Encoding**: Text to numeric conversion with one-hot encoding
- **TensorFlow Pipeline**: Optimized data pipeline with batching and caching
- **Feature Extraction**: MobileNetV2-based feature extraction in batches

### 5. Recommendation System
- **User Input Collection**: Gender, subcategory, and usage preferences
- **Similarity Calculation**: Cosine similarity between feature vectors
- **Product Filtering**: Preference-based filtering and ranking
- **Result Display**: Visual presentation of recommended items

## 🏗️ Model Architecture

### MobileNetV2 Architecture
The system utilizes MobileNetV3 for efficient feature extraction:
![image](https://github.com/user-attachments/assets/df3b36d5-a1a0-4118-b827-e68dc415ec72)


**Key Components:**
- **Input Layer**: 224×224×3 RGB images
- **Initial Conv Layer**: 3×3 kernel, stride 2, followed by BatchNorm and ReLU6
- **Bottleneck Blocks**: 
  - 1×1 expansion convolution
  - 3×3 depthwise convolution
  - 1×1 projection convolution
  - Residual connections when dimensions match
- **Global Average Pooling**: Spatial dimension reduction
- **Feature Output**: Dense feature vectors for similarity comparison

## 💻 Usage

1. **Run the Jupyter Notebook**
   ```bash
   jupyter notebook project_ai.ipynb
   ```

2. **Follow the Interactive Interface**
   - Select gender preference (Men/Women)
   - Choose subcategory from available options
   - Specify usage context (Casual, Party, Sports, etc.)

3. **View Recommendations**
   - System displays top 5 similar products
   - Visual grid showing recommended items with style information

## 📸 Sample Output

The system provides visual recommendations showing:
- Product images in a grid layout
- Style information for each recommendation
- Similarity scores and product details

**Example Categories:**
- Scarves (Women's Casual)
- Bottomwear (Men's Casual) 
- Dresses (Women's Party)
- Topwear (Men's Casual)

<img width="977" height="914" alt="image" src="https://github.com/user-attachments/assets/b9fb17ae-704d-4d06-8931-fa43e526ac2b" />



<img width="900" height="826" alt="image" src="https://github.com/user-attachments/assets/b5c23f0c-9afc-4e63-8823-c38c7aa56c3a" />


## 🔧 Key Functions

- `preprocess_image()`: Image loading and preprocessing
- `extract_features_batch()`: Batch feature extraction with progress tracking
- `get_user_preferences()`: Interactive user input collection
- `recommend_similar_items()`: Similarity-based recommendation engine
- `display_recommendations()`: Visual presentation of results

## 📈 Performance Optimizations

- **Batch Processing**: Processes 2000 images per batch for memory efficiency
- **Progress Tracking**: Real-time progress monitoring with tqdm
- **Error Handling**: Robust handling of corrupted or missing images
- **Pipeline Optimization**: TensorFlow data pipeline with prefetching and caching

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Esraa Mohammed** - [GitHub Profile](https://github.com/esraamhmd)

## 🙏 Acknowledgments

- Kaggle for providing the fashion dataset
- TensorFlow team for the MobileNetV2 pre-trained model
- Open source community for the various libraries used

## 📞 Contact

For questions or suggestions, please open an issue on GitHub or contact the repository owner.

---

⭐ **Don't forget to star this repository if you found it helpful!**
