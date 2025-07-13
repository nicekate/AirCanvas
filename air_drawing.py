import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import math

class AirDrawing:
    def __init__(self):
        # 初始化MediaPipe手部检测
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # 画布设置
        self.canvas = None
        self.drawing_points = deque(maxlen=1024)
        
        # 颜色设置
        self.colors = {
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'cyan': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0)
        }
        self.current_color = self.colors['red']
        self.color_names = list(self.colors.keys())
        self.current_color_index = 0
        
        # 绘画状态
        self.is_drawing = False
        self.prev_point = None
        self.brush_size = 5
        
        # 手势识别参数
        self.gesture_threshold = 0.1
        self.color_change_cooldown = 0
        self.clear_gesture_cooldown = 0
        
    def get_finger_positions(self, landmarks):
        """获取手指关键点位置"""
        # 食指尖端 (landmark 8)
        index_tip = landmarks[8]
        # 中指尖端 (landmark 12)
        middle_tip = landmarks[12]
        # 拇指尖端 (landmark 4)
        thumb_tip = landmarks[4]
        # 无名指尖端 (landmark 16)
        ring_tip = landmarks[16]
        # 小指尖端 (landmark 20)
        pinky_tip = landmarks[20]
        
        return {
            'index': index_tip,
            'middle': middle_tip,
            'thumb': thumb_tip,
            'ring': ring_tip,
            'pinky': pinky_tip
        }
    
    def is_finger_up(self, landmarks, finger_tip_id, finger_pip_id):
        """检查手指是否伸直"""
        return landmarks[finger_tip_id].y < landmarks[finger_pip_id].y
    
    def detect_gesture(self, landmarks):
        """检测手势 - 改进版本"""
        fingers = []

        # 拇指检测 (更严格的判断)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]

        # 拇指伸直判断：尖端要明显高于关节
        thumb_up = thumb_tip.y < thumb_ip.y - 0.02 and thumb_tip.y < thumb_mcp.y - 0.02
        fingers.append(thumb_up)

        # 其他四个手指 - 更严格的判断
        finger_data = [
            (8, 6, 5),   # 食指: tip, pip, mcp
            (12, 10, 9), # 中指: tip, pip, mcp
            (16, 14, 13), # 无名指: tip, pip, mcp
            (20, 18, 17)  # 小指: tip, pip, mcp
        ]

        for tip_id, pip_id, mcp_id in finger_data:
            tip = landmarks[tip_id]
            pip = landmarks[pip_id]
            mcp = landmarks[mcp_id]

            # 手指伸直判断：尖端要明显高于两个关节
            finger_up = (tip.y < pip.y - 0.015 and tip.y < mcp.y - 0.02)
            fingers.append(finger_up)

        return fingers
    
    def get_distance(self, point1, point2):
        """计算两点之间的距离"""
        return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def process_frame(self, frame):
        """处理每一帧"""
        height, width, _ = frame.shape
        
        if self.canvas is None:
            self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 翻转图像以获得镜像效果
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 检测手部
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # 获取手指位置
                fingers = self.get_finger_positions(hand_landmarks.landmark)
                
                # 检测手势
                finger_states = self.detect_gesture(hand_landmarks.landmark)
                fingers_up_count = sum(finger_states)

                # 获取食指尖端位置
                index_x = int(fingers['index'].x * width)
                index_y = int(fingers['index'].y * height)

                # 手势识别逻辑 - 更精确的判断
                if fingers_up_count == 1 and finger_states[1]:
                    # 只有食指伸直 - 绘画模式
                    if self.prev_point is not None:
                        cv2.line(self.canvas, self.prev_point, (index_x, index_y),
                                self.current_color, self.brush_size)
                    self.prev_point = (index_x, index_y)
                    self.is_drawing = True

                    # 在实时画面上显示绘画点
                    cv2.circle(frame, (index_x, index_y), self.brush_size,
                              self.current_color, -1)

                elif (fingers_up_count == 2 and finger_states[1] and finger_states[2] and
                      self.color_change_cooldown <= 0):
                    # 食指和中指伸直 - 切换颜色
                    self.change_color()
                    self.color_change_cooldown = 60  # 增加冷却时间
                    self.prev_point = None
                    self.is_drawing = False

                elif (fingers_up_count >= 4 and all(finger_states[1:]) and
                      self.clear_gesture_cooldown <= 0):
                    # 至少4个手指伸直（除拇指外全部） - 清除画布
                    self.clear_canvas()
                    self.clear_gesture_cooldown = 90  # 更长的冷却时间
                    self.prev_point = None
                    self.is_drawing = False

                else:
                    # 其他手势 - 停止绘画
                    self.prev_point = None
                    self.is_drawing = False
        
        # 减少冷却时间
        if self.color_change_cooldown > 0:
            self.color_change_cooldown -= 1
        if self.clear_gesture_cooldown > 0:
            self.clear_gesture_cooldown -= 1
        
        # 将画布叠加到实时画面上
        frame = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)

        # 显示当前颜色和说明
        if results.multi_hand_landmarks:
            # 如果检测到手部，显示手势信息
            finger_states = self.detect_gesture(results.multi_hand_landmarks[0].landmark)
            fingers_count = sum(finger_states)
            self.draw_ui(frame, finger_states, fingers_count)
        else:
            # 没有检测到手部
            self.draw_ui(frame)

        return frame
    
    def change_color(self):
        """切换颜色"""
        self.current_color_index = (self.current_color_index + 1) % len(self.color_names)
        color_name = self.color_names[self.current_color_index]
        self.current_color = self.colors[color_name]
        print(f"颜色切换为: {color_name}")
    
    def clear_canvas(self):
        """清除画布"""
        if self.canvas is not None:
            self.canvas.fill(0)
        print("画布已清除")
    
    def draw_ui(self, frame, finger_states=None, fingers_count=0):
        """绘制用户界面"""
        height, width = frame.shape[:2]

        # 显示当前颜色
        color_name = self.color_names[self.current_color_index]
        cv2.putText(frame, f"Color: {color_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, self.current_color, 2)

        # 显示手指检测状态
        if finger_states is not None:
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            cv2.putText(frame, f"Fingers: {fingers_count}/5", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # 显示每个手指的状态
            for i, (name, state) in enumerate(zip(finger_names, finger_states)):
                color = (0, 255, 0) if state else (0, 0, 255)
                cv2.putText(frame, f"{name}: {'UP' if state else 'DOWN'}",
                           (10, 100 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 显示操作说明
        instructions = [
            "1 finger (Index): Draw",
            "2 fingers (Index+Middle): Change color",
            "4+ fingers: Clear canvas",
            "Press 'q' to quit"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, height - 100 + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 显示颜色选择器
        for i, (name, color) in enumerate(self.colors.items()):
            x = width - 150
            y = 50 + i * 30
            cv2.rectangle(frame, (x, y), (x + 20, y + 20), color, -1)
            if i == self.current_color_index:
                cv2.rectangle(frame, (x - 2, y - 2), (x + 22, y + 22), (255, 255, 255), 2)
            cv2.putText(frame, name, (x + 30, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def main():
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 创建空中绘画对象
    air_drawing = AirDrawing()
    
    print("🎨 AirCanvas 已启动!")
    print("手势说明:")
    print("- 只伸出食指: 绘画")
    print("- 伸出食指和中指: 切换颜色")
    print("- 伸出所有手指: 清除画布")
    print("- 按 'q' 键退出")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 处理帧
        processed_frame = air_drawing.process_frame(frame)
        
        # 显示结果
        cv2.imshow('AirCanvas - 空中画布', processed_frame)
        
        # 检查退出条件
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            air_drawing.clear_canvas()
        elif key == ord('n'):
            air_drawing.change_color()
    
    # 清理资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
