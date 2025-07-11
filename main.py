import cv2 as cv
from line_detection import LineDetector
from tqdm import tqdm


def detect_offside(videopath):
    """
    In the given input video, perform the real time offside detection.

    videopath: filename for the video
    """
    # Create a VideoCapture object
    cap = cv.VideoCapture(videopath)

    # Check if the video capture object is opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        exit()

    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    frame_count = 0
    with tqdm(total=1900) as pbar:
        while cap.isOpened() and frame_count < 1900:
            ret, frame = cap.read()
            if not ret:
                break  # Break the loop if no frame is read (end of video)

            frame_count += 1

            # Update progress bar
            pbar.update(1)

    
    # Get the video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Choose the codec as needed
    out = cv.VideoWriter('output/output_video.mp4', fourcc, fps, (width, height))

    # Process each frame in the video

    with tqdm(total=total_frames - 1900) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()  # Read a frame
            if not ret:
                break  # Break the loop if no frame is read (end of video)
            
            line_detector = LineDetector(frame)
            segmented_img = line_detector.segment_playing_area()
            _, _, all_lines, _ = line_detector.detect_offside_lines(segmented_img)
            vanishing_point = line_detector.compute_vanishing_point(all_lines)
            lines_image = line_detector.draw_detected_lines(all_lines)
            processed_frame = line_detector.draw_pose_lines(vanishing_point, lines_image)
            out.write(processed_frame)

            pbar.update(1)
            

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv.destroyAllWindows()


def main():
    videopath = 'dataset/video_city_westham.mov'

    detect_offside(videopath)

if __name__ == "__main__":
    main()


