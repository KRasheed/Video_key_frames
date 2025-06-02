# 🎬 Adaptive Keyframe Extractor

A powerful Streamlit web application that intelligently extracts key frames from videos using adaptive computer vision algorithms. Perfect for video summarization, content analysis, scene detection, and thumbnail generation.

## ✨ Features

- **🧠 Intelligent Detection**: Uses histogram analysis and frame differencing to identify scene changes
- **⚙️ Configurable Parameters**: Adjustable sensitivity thresholds via intuitive sidebar controls
- **📊 Visual Analysis**: Timeline plots and difference score graphs for extraction insights
- **🖼️ Grid Display**: Clean keyframe preview with timestamps and frame numbers
- **💾 Batch Download**: Export all keyframes as a ZIP file with organized naming

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/keyframe-extraction.git
   cd keyframe-extraction
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```
## 🎯 Usage

### Basic Workflow

1. **Upload Video**: Support for MP4, AVI, MOV, MKV, WMV, FLV formats
2. **Configure Parameters**: Adjust extraction sensitivity using sidebar controls
3. **Extract Keyframes**: Click the "Extract Keyframes" button to start processing
4. **Analyze Results**: Review timeline plots and keyframe distribution
5. **Download**: Export keyframes as a ZIP file for further use

### Parameter Configuration

| Parameter | Range | Description |
|-----------|-------|-------------|
| **Histogram Threshold** | 0.1 - 0.9 | Lower values = more sensitive to color changes |
| **Frame Difference Threshold** | 10 - 80 | Higher values = less sensitive to motion |
| **Minimum Frame Interval** | 5 - 60 | Minimum frames between keyframes |
| **Maximum Keyframes** | 10 - 500 | Maximum number of keyframes to extract |

## 🔬 Algorithm Details

The adaptive keyframe extraction uses a multi-layered approach:

### 1. **Histogram Analysis**
- Converts frames to HSV color space for better color representation
- Calculates 3D histograms for hue, saturation, and value channels
- Uses correlation coefficient to measure color similarity

### 2. **Frame Differencing**
- Converts frames to grayscale for structural comparison
- Calculates absolute pixel differences between consecutive frames
- Normalizes difference scores as percentages

### 3. **Adaptive Decision Making**
- Combines histogram and frame difference metrics
- Applies configurable thresholds for keyframe detection
- Ensures temporal spacing with minimum interval filtering

### 4. **Performance Optimization**
- Automatic frame resizing (max 640px width) for faster processing
- Progressive loading with real-time progress updates
- Memory-efficient processing for longer videos

## 📊 Performance Guidelines

### Recommended Video Lengths

| Duration | Status | Expected Processing Time |
|----------|--------|-------------------------|
| **< 5 minutes** | ✅ Optimal | 10-30 seconds |
| **5-15 minutes** | ⚠️ Acceptable | 1-5 minutes |
| **> 15 minutes** | ❌ Not Recommended | Memory issues likely |

### Memory Considerations

- **Video Resolution**: Higher resolution = more memory usage
- **Video Length**: Linear scaling with duration
- **Keyframe Count**: Each extracted frame consumes memory
- **File Size Limit**: Recommended < 100MB for smooth experience


## 🎨 Output Format

### Keyframe Files
- **Format**: JPEG (95% quality)
- **Naming**: `keyframe_001_frame_000123.jpg`
- **Organization**: ZIP archive with sequential numbering

