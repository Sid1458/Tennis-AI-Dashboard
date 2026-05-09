"""
Standalone pipeline runner — called as subprocess from streamlit_app.py
Usage: python pipeline_runner.py <input_video_path> <output_dir>
Prints: PROGRESS:<0-100>:<message>  and  OUTPUT_VIDEO:<path>  STATS_FILE:<path>
"""
import os, sys, json
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

def prog(n, msg=""):
    print(f"PROGRESS:{n}:{msg}", flush=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: pipeline_runner.py <input_video> <output_dir>", file=sys.stderr)
        sys.exit(1)

    input_path, output_dir = sys.argv[1], sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    import cv2, pandas as pd
    from copy import deepcopy
    from utils import (read_video, save_video, measure_distance,
                       draw_player_stats, convert_pixel_distance_to_meters)
    import constants
    from trackers import PlayerTracker, BallTracker
    from court_line_detector import CourtLineDetector
    from mini_court import MiniCourt

    prog(5,  "Reading video frames")
    video_frames = read_video(input_path)

    prog(10, "Loading AI models")
    player_tracker = PlayerTracker(model_path='yolov8x.pt')
    ball_tracker   = BallTracker(model_path='models/yolo5_last.pt')

    stub_player = "tracker_stubs/player_detections.pkl"
    stub_ball   = "tracker_stubs/ball_detections.pkl"

    # Smart stub check: stubs only work if frame count matches the uploaded video
    force_full = sys.argv[3] == "full" if len(sys.argv) > 3 else False
    stubs_exist = os.path.exists(stub_player) and os.path.exists(stub_ball)

    if stubs_exist and not force_full:
        import pickle
        with open(stub_player, "rb") as f:
            stub_len = len(pickle.load(f))
        if stub_len != len(video_frames):
            prog(18, f"⚠️ Stub frame count ({stub_len}) ≠ video frames ({len(video_frames)}) → switching to full inference")
            use_stubs = False
        else:
            use_stubs = True
    else:
        use_stubs = False

    if use_stubs:
        prog(20, "Loading pre-computed detections (stub mode)")
        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path=stub_player)
        prog(35, "Loading ball detections")
        ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path=stub_ball)
    else:
        prog(20, "Detecting players (full inference)")
        player_detections = player_tracker.detect_frames(video_frames, read_from_stub=False)
        prog(40, "Tracking ball (full inference)")
        ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=False)

    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)
    prog(50, "Ball interpolation done")

    prog(55, "Detecting court lines")
    court_detector  = CourtLineDetector("models/keypoints_model.pth")
    court_keypoints = court_detector.predict(video_frames[0])
    prog(65, "Court lines detected")

    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Remap arbitrary YOLO track IDs → 1 and 2 (mini_court expects exactly these keys)
    all_ids = sorted({pid for frame in player_detections for pid in frame.keys()})
    if len(all_ids) >= 2:
        id_map = {all_ids[0]: 1, all_ids[1]: 2}
        player_detections = [
            {id_map.get(pid, pid): bbox for pid, bbox in frame.items()}
            for frame in player_detections
        ]

    mini_court = MiniCourt(video_frames[0])
    prog(70, "Mini court ready")

    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)
    player_mini, ball_mini = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)
    prog(78, "Shot analysis done")

    prog(80, "Computing statistics")
    stats = [{'frame_num':0,'player_1_number_of_shots':0,'player_1_total_shot_speed':0,
               'player_1_last_shot_speed':0,'player_1_total_player_speed':0,'player_1_last_player_speed':0,
               'player_2_number_of_shots':0,'player_2_total_shot_speed':0,
               'player_2_last_shot_speed':0,'player_2_total_player_speed':0,'player_2_last_player_speed':0}]

    for i in range(len(ball_shot_frames)-1):
        sf, ef = ball_shot_frames[i], ball_shot_frames[i+1]
        t = (ef-sf)/24
        d_ball = measure_distance(ball_mini[sf][1], ball_mini[ef][1])
        ball_speed = convert_pixel_distance_to_meters(d_ball, constants.DOUBLE_LINE_WIDTH,
                     mini_court.get_width_of_mini_court()) / t * 3.6 if t>0 else 0
        pp = player_mini[sf]
        shot_by = min(pp.keys(), key=lambda pid: measure_distance(pp[pid], ball_mini[sf][1]))
        opp = 1 if shot_by==2 else 2
        d_opp = measure_distance(player_mini[sf][opp], player_mini[ef][opp])
        opp_speed = convert_pixel_distance_to_meters(d_opp, constants.DOUBLE_LINE_WIDTH,
                    mini_court.get_width_of_mini_court()) / t * 3.6 if t>0 else 0
        cur = deepcopy(stats[-1]); cur['frame_num'] = sf
        cur[f'player_{shot_by}_number_of_shots'] += 1
        cur[f'player_{shot_by}_total_shot_speed'] += ball_speed
        cur[f'player_{shot_by}_last_shot_speed']  = ball_speed
        cur[f'player_{opp}_total_player_speed']   += opp_speed
        cur[f'player_{opp}_last_player_speed']     = opp_speed
        stats.append(cur)

    df = pd.DataFrame(stats)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    
    # Phase 2: Add player mini-court coordinates for every frame (for Heatmap/Dominance)
    p1_x, p1_y, p2_x, p2_y = [], [], [], []
    for frame_idx, p_dict in enumerate(player_mini):
        p1 = p_dict.get(1, (0, 0))
        p2 = p_dict.get(2, (0, 0))
        p1_x.append(p1[0]); p1_y.append(p1[1])
        p2_x.append(p2[0]); p2_y.append(p2[1])
        
    frames_df['player_1_x'] = p1_x
    frames_df['player_1_y'] = p1_y
    frames_df['player_2_x'] = p2_x
    frames_df['player_2_y'] = p2_y

    df = pd.merge(frames_df, df, on='frame_num', how='left').ffill()
    df['player_1_average_shot_speed']   = df['player_1_total_shot_speed']   / df['player_1_number_of_shots']
    df['player_2_average_shot_speed']   = df['player_2_total_shot_speed']   / df['player_2_number_of_shots']
    df['player_1_average_player_speed'] = df['player_1_total_player_speed'] / df['player_2_number_of_shots']
    df['player_2_average_player_speed'] = df['player_2_total_player_speed'] / df['player_1_number_of_shots']
    prog(87, "Stats computed")

    prog(88, "Rendering annotated video")
    out_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    out_frames = ball_tracker.draw_bboxes(out_frames, ball_detections)
    out_frames = court_detector.draw_keypoints_on_video(out_frames, court_keypoints)
    out_frames = mini_court.draw_mini_court(out_frames)
    out_frames = mini_court.draw_points_on_mini_court(out_frames, player_mini)
    out_frames = mini_court.draw_points_on_mini_court(out_frames, ball_mini, color=(0,255,255))
    out_frames = draw_player_stats(out_frames, df)
    for idx, frame in enumerate(out_frames):
        cv2.putText(frame, f"Frame: {idx}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Save as H.264 MP4 (browser-compatible) via imageio + ffmpeg
    import imageio
    out_video = os.path.join(output_dir, "output.mp4")
    fps_out = 24
    with imageio.get_writer(out_video, fps=fps_out, codec='libx264',
                            quality=7, macro_block_size=8) as writer:
        for frame in out_frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    stats_file = os.path.join(output_dir, "stats.csv")
    df.to_csv(stats_file, index=False)

    prog(100, "Done")
    print(f"OUTPUT_VIDEO:{out_video}", flush=True)
    print(f"STATS_FILE:{stats_file}", flush=True)
    print(f"USED_STUBS:{use_stubs}", flush=True)

if __name__ == "__main__":
    main()
