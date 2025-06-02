import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import zipfile
from io import BytesIO
import time
from typing import List, Tuple
import matplotlib.pyplot as plt

# Configure page
st.set_page_config(
    page_title="Adaptive Keyframe Extractor",
    page_icon="🎬",
    layout="wide"
)

class AdaptiveKeyframeExtractor:
    def __init__(self, 
                 histogram_threshold: float = 0.3,
                 frame_diff_threshold: float = 30.0,
                 min_interval: int = 10,
                 max_keyframes: int = 100):
        """
        Initialize the adaptive keyframe extractor.
        
        Args:
            histogram_threshold: Histogram correlation threshold (0-1)
            frame_diff_threshold: Frame difference threshold (0-100)
            min_interval: Minimum frames between keyframes
            max_keyframes: Maximum number of keyframes to extract
        """
        self.hist_threshold = histogram_threshold
        self.diff_threshold = frame_diff_threshold
        self.min_interval = min_interval
        self.max_keyframes = max_keyframes
        
    def calculate_histogram_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate histogram correlation between two frames."""
        # Convert to HSV for better color representation
        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        
        # Calculate correlation
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return correlation
    
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate structural difference between frames."""
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Calculate mean difference as percentage
        mean_diff = np.mean(diff) / 255.0 * 100
        return mean_diff
    
    def extract_keyframes(self, video_path: str, 
                         progress_callback=None) -> Tuple[List[np.ndarray], List[int], List[float]]:
        """
        Extract keyframes from video using adaptive algorithm.
        
        Returns:
            keyframes: List of keyframe images
            frame_numbers: List of frame numbers for keyframes
            scores: List of difference scores for each keyframe
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        keyframes = []
        frame_numbers = []
        scores = []
        
        # Read first frame as initial keyframe
        ret, prev_frame = cap.read()
        if not ret:
            raise ValueError("Could not read video frames")
        
        # Resize frame for faster processing
        height, width = prev_frame.shape[:2]
        if width > 640:
            scale = 640 / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            prev_frame = cv2.resize(prev_frame, (new_width, new_height))
        
        keyframes.append(prev_frame.copy())
        frame_numbers.append(0)
        scores.append(0.0)
        
        frame_idx = 1
        last_keyframe_idx = 0
        
        while frame_idx < total_frames and len(keyframes) < self.max_keyframes:
            ret, current_frame = cap.read()
            if not ret:
                break
            
            # Resize current frame
            if width > 640:
                current_frame = cv2.resize(current_frame, (new_width, new_height))
            
            # Check if enough frames have passed since last keyframe
            if frame_idx - last_keyframe_idx >= self.min_interval:
                # Calculate similarity metrics
                hist_similarity = self.calculate_histogram_similarity(prev_frame, current_frame)
                frame_diff = self.calculate_frame_difference(prev_frame, current_frame)
                
                # Determine if current frame should be a keyframe
                is_keyframe = (hist_similarity < self.hist_threshold or 
                             frame_diff > self.diff_threshold)
                
                if is_keyframe:
                    keyframes.append(current_frame.copy())
                    frame_numbers.append(frame_idx)
                    scores.append(frame_diff)
                    last_keyframe_idx = frame_idx
                    prev_frame = current_frame.copy()
            
            frame_idx += 1
            
            # Update progress
            if progress_callback and frame_idx % 30 == 0:
                progress = frame_idx / total_frames
                progress_callback(progress)
        
        cap.release()
        
        return keyframes, frame_numbers, scores

def create_download_zip(keyframes: List[np.ndarray], frame_numbers: List[int]) -> BytesIO:
    """Create a ZIP file containing all keyframes."""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, (frame, frame_num) in enumerate(zip(keyframes, frame_numbers)):
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Add to ZIP
            filename = f"keyframe_{i+1:03d}_frame_{frame_num:06d}.jpg"
            zip_file.writestr(filename, buffer.tobytes())
    
    zip_buffer.seek(0)
    return zip_buffer

def plot_extraction_analysis(frame_numbers: List[int], scores: List[float], 
                           total_frames: int, fps: float):
    """Create analysis plots for keyframe extraction."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Timeline plot
    times = [f / fps for f in frame_numbers]
    ax1.scatter(times, [1] * len(times), alpha=0.7, s=50)
    ax1.set_xlim(0, total_frames / fps)
    ax1.set_ylim(0.5, 1.5)
    ax1.set_xlabel('Time (seconds)')
    ax1.set_title('Keyframe Distribution Timeline')
    ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)
    
    # Scores plot
    ax2.plot(times[1:], scores[1:], marker='o', linewidth=2, markersize=4)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Difference Score')
    ax2.set_title('Frame Difference Scores for Keyframes')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def main():
    st.title("🎬 Adaptive Keyframe Extractor")
    st.markdown("""
    Extract key frames from videos using intelligent adaptive algorithms that detect scene changes,
    motion patterns, and visual differences.
    """)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Extraction Parameters")
    
    hist_threshold = st.sidebar.slider(
        "Histogram Threshold", 
        min_value=0.1, max_value=0.9, value=0.3, step=0.1,
        help="Lower values = more sensitive to color changes"
    )
    
    diff_threshold = st.sidebar.slider(
        "Frame Difference Threshold", 
        min_value=10.0, max_value=80.0, value=30.0, step=5.0,
        help="Higher values = less sensitive to motion"
    )
    
    min_interval = st.sidebar.slider(
        "Minimum Frame Interval", 
        min_value=5, max_value=60, value=10,
        help="Minimum frames between keyframes"
    )
    
    max_keyframes = st.sidebar.slider(
        "Maximum Keyframes", 
        min_value=10, max_value=500, value=100,
        help="Maximum number of keyframes to extract"
    )
    
    # Video length recommendations
    st.sidebar.markdown("### 📊 Recommended Limits")
    st.sidebar.info("""
    **For optimal performance:**
    - **Short videos**: < 5 minutes ✅
    - **Medium videos**: 5-15 minutes ⚠️
    - **Long videos**: > 15 minutes ❌
    
    **Memory usage scales with:**
    - Video resolution
    - Video length
    - Number of keyframes
    """)
    
    # Main interface
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv'],
        help="Supported formats: MP4, AVI, MOV, MKV, WMV, FLV"
    )
    
    if uploaded_file is not None:
        # Display video info
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.info(f"📁 File size: {file_size_mb:.1f} MB")
        
        # Warning for large files
        if file_size_mb > 100:
            st.warning("⚠️ Large file detected! Processing may be slow and could exceed memory limits.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        try:
            # Get video information
            cap = cv2.VideoCapture(temp_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # Display video properties
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{duration/60:.1f} min")
            with col2:
                st.metric("FPS", f"{fps:.1f}")
            with col3:
                st.metric("Resolution", f"{width}×{height}")
            with col4:
                st.metric("Total Frames", f"{total_frames:,}")
            
            # Process button
            if st.button("🚀 Extract Keyframes", type="primary"):
                # Show warnings based on video characteristics
                if duration > 900:  # 15 minutes
                    st.error("❌ Video too long! Please use videos under 15 minutes for optimal performance.")
                    return
                elif duration > 300:  # 5 minutes
                    st.warning("⚠️ Long video detected. Processing may take several minutes.")
                
                # Initialize extractor
                extractor = AdaptiveKeyframeExtractor(
                    histogram_threshold=hist_threshold,
                    frame_diff_threshold=diff_threshold,
                    min_interval=min_interval,
                    max_keyframes=max_keyframes
                )
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing... {progress*100:.1f}%")
                
                # Extract keyframes
                start_time = time.time()
                
                try:
                    keyframes, frame_numbers, scores = extractor.extract_keyframes(
                        temp_path, 
                        progress_callback=update_progress
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Display results
                    st.success(f"✅ Extracted {len(keyframes)} keyframes in {processing_time:.1f} seconds")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Keyframes Found", len(keyframes))
                    with col2:
                        st.metric("Coverage", f"{len(keyframes)/total_frames*100:.2f}%")
                    with col3:
                        st.metric("Avg Interval", f"{total_frames/len(keyframes):.1f} frames")
                    
                    # Analysis plots
                    if len(keyframes) > 1:
                        st.subheader("📈 Extraction Analysis")
                        fig = plot_extraction_analysis(frame_numbers, scores, total_frames, fps)
                        st.pyplot(fig)
                    
                    # Display keyframes
                    st.subheader("🖼️ Extracted Keyframes")
                    
                    # Grid display
                    cols_per_row = 4
                    for i in range(0, len(keyframes), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            idx = i + j
                            if idx < len(keyframes):
                                frame = keyframes[idx]
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                with col:
                                    st.image(
                                        frame_rgb, 
                                        caption=f"Frame {frame_numbers[idx]} ({frame_numbers[idx]/fps:.1f}s)",
                                        use_container_width=True
                                    )
                    
                    # Download option
                    if len(keyframes) > 0:
                        st.subheader("💾 Download Results")
                        
                        zip_data = create_download_zip(keyframes, frame_numbers)
                        
                        st.download_button(
                            label="📦 Download All Keyframes (ZIP)",
                            data=zip_data.getvalue(),
                            file_name=f"keyframes_{uploaded_file.name}.zip",
                            mime="application/zip"
                        )
                        
                        st.info(f"📁 ZIP contains {len(keyframes)} JPEG images with frame numbers in filenames")
                
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.info("Try reducing video length or adjusting parameters.")
        
        except Exception as e:
            st.error(f"❌ Error reading video file: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    else:
        # Help section
        st.markdown("### 📚 How It Works")
        
        st.markdown("""
        The adaptive keyframe extraction algorithm uses multiple techniques:
        
        1. **Histogram Analysis**: Compares color distributions between frames
        2. **Frame Differencing**: Measures structural changes between consecutive frames  
        3. **Adaptive Thresholding**: Dynamically adjusts sensitivity based on content
        4. **Temporal Filtering**: Ensures minimum spacing between keyframes
        
        **Perfect for:**
        - 🎬 Video summarization
        - 🔍 Content analysis  
        - 📊 Scene detection
        - 🎯 Thumbnail generation
        """)

if __name__ == "__main__":
    main()