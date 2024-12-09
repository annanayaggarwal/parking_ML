import cv2

def draw_results(frame, spots, results):
    for spot_id, (x1, y1, x2, y2) in spots.items():
        status = results[spot_id]['status']
        confidence = results[spot_id]['confidence']
        
        color = (0, 255, 0) if status == 'empty' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, 
                   f"{spot_id}: {status} ({confidence:.2f})",
                   (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5,
                   color,
                   2)
    return frame