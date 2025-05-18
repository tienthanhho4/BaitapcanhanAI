8-Puzzle Solver - Bài tập Cá nhân AI
1. Mục tiêu
Trong đồ án cá nhân, các nhóm thuật toán tìm kiếm trong Trí tuệ nhân tạo được nghiên cứu và áp dụng nhằm giải quyết bài toán 8-puzzle – một bài toán cổ điển thể hiện rõ đặc trưng của việc tìm kiếm lời giải trong không gian trạng thái. Cụ thể, đề tài tập trung vào 6 nhóm thuật toán chính:

Thuật toán tìm kiếm không có thông tin (Uninformed Search): Như BFS, DFS, IDS và UCS, giúp khảo sát khả năng tìm lời giải khi không có thông tin định hướng.
Thuật toán tìm kiếm có thông tin (Informed Search): Như A*, IDA* và Greedy Best-First Search, sử dụng heuristic để tối ưu hóa hiệu quả tìm kiếm.
Tìm kiếm cục bộ (Local Search): Như Hill Climbing, Steepest Ascent Hill Climbing, Simple Hill Climbing, Simulated Annealing, Stochastic Hill Climbing và Beam Search, tập trung vào việc cải thiện nghiệm cục bộ mà không cần duy trì toàn bộ không gian trạng thái.
Tìm kiếm trong môi trường phức tạp (Searching in Complex Environments): Như AND-OR Graph Search, Searching for a partially observation, Sensorless, mở rộng khả năng ứng dụng sang các bài toán có tính động và không chắc chắn, định hướng cho các nghiên cứu nâng cao.
Bài toán thỏa mãn ràng buộc (Constraint Satisfaction Problems - CSP): Như Forward-Checking, Backtracking, nhằm khảo sát khả năng biểu diễn 8-puzzle dưới dạng hệ thống ràng buộc logic.
Học tăng cường (Reinforcement Learning): Cụ thể là thuật toán Q-learning, cho phép tác nhân học cách giải quyết bài toán thông qua việc tương tác với môi trường.

Việc triển khai và so sánh các nhóm thuật toán này không chỉ giúp đánh giá hiệu quả của từng phương pháp mà còn mở ra các hướng tiếp cận đa dạng, góp phần làm phong phú thêm ứng dụng của Trí tuệ nhân tạo trong giải quyết các bài toán tìm kiếm.
2. Nội dung
2.1. Thuật toán tìm kiếm không có thông tin (Uninformed Search Algorithms)
Một bài toán tìm kiếm trong trí tuệ nhân tạo thường bao gồm các thành phần chính sau:

Không gian trạng thái (State space): Tập hợp tất cả các trạng thái có thể có của bài toán.
Trạng thái khởi đầu (Initial state): Trạng thái bắt đầu của bài toán.
Trạng thái đích (Goal state): Trạng thái hoặc tập các trạng thái mà ta muốn tìm đến.
Hàm chuyển đổi (Transition function): Các phép biến đổi từ trạng thái này sang trạng thái khác.
Hàm kiểm tra trạng thái đích (Goal test): Kiểm tra xem trạng thái hiện tại có phải là trạng thái đích không.
Chi phí (Cost function): Chi phí để đi từ trạng thái này sang trạng thái khác (nếu có).
Solution (giải pháp): Là một chuỗi các hành động hoặc trạng thái từ trạng thái khởi đầu đến trạng thái đích thỏa mãn bài toán tìm kiếm.

Các thuật toán triển khai:

BFS (Breadth-First Search)
DFS (Depth-First Search)
UCS (Uniform-Cost Search)
IDS (Iterative Deepening Search)

Nhận xét:

DFS: Duyệt sâu vào nhánh trước, tốn ít bộ nhớ nhưng dễ rơi vào vòng lặp vô hạn và không đảm bảo tìm giải pháp tối ưu; không phù hợp cho không gian trạng thái lớn như 8-puzzle.
BFS: Đảm bảo tìm được giải pháp ngắn nhất nhưng tốn rất nhiều bộ nhớ và thời gian khi không gian trạng thái lớn, dễ bị bùng nổ tổ hợp trong 8-puzzle.
UCS: Tương tự BFS nhưng xét chi phí đường đi, đảm bảo tìm giải pháp tối ưu theo chi phí, tuy nhiên cũng rất tốn bộ nhớ và thời gian trong bài toán 8-puzzle.
IDS: Kết hợp ưu điểm của DFS và BFS, tiết kiệm bộ nhớ hơn BFS, tránh vòng lặp của DFS, nhưng thường chậm hơn do phải lặp lại tìm kiếm nhiều lần; vẫn chưa hiệu quả bằng các thuật toán heuristic trong 8-puzzle.
Kết luận: Các thuật toán không thông tin này đều có hạn chế về hiệu suất khi áp dụng cho bài toán 8-puzzle do không sử dụng thông tin hướng dẫn, dẫn đến tốn nhiều thời gian và bộ nhớ.

2.2. Thuật toán tìm kiếm có thông tin (Informed Search Algorithms)
Các thành phần cơ bản của bài toán tìm kiếm:

Trạng thái ban đầu (Initial state): Trạng thái xuất phát của bài toán.
Trạng thái đích (Goal state): Trạng thái hoặc tập trạng thái mà ta cần tìm đến.
Hành động (Actions): Các phép biến đổi để chuyển từ trạng thái này sang trạng thái khác.
Hàm chi phí (Cost function): Chi phí thực hiện mỗi hành động hoặc di chuyển giữa các trạng thái.
Hàm kiểm tra trạng thái đích (Goal test): Kiểm tra xem trạng thái hiện tại có phải là trạng thái đích không.
Solution (giải pháp): Chuỗi các hành động hoặc trạng thái từ trạng thái ban đầu đến trạng thái đích, thỏa mãn yêu cầu của bài toán tìm kiếm.

Các thuật toán triển khai:

A*
IDA*
Greedy Best-First Search

Nhận xét:

A*: Là lựa chọn hàng đầu cho bài toán 8-puzzle nhờ khả năng tìm lời giải tối ưu với hiệu suất tốt khi sử dụng hàm heuristic phù hợp.
Greedy Best-First Search: Nhanh nhưng không tối ưu, dễ mắc sai lầm trong không gian trạng thái phức tạp.
IDA*: Là giải pháp thay thế cho A* khi bộ nhớ hạn chế, vẫn đảm bảo tính tối ưu nhưng đổi lại thời gian chạy có thể lâu hơn.
Kết luận: Các thuật toán tìm kiếm có sử dụng thông tin (heuristic) giúp giảm đáng kể số trạng thái cần duyệt so với các thuật toán tìm kiếm không thông tin, đặc biệt trong các bài toán phức tạp như trò chơi 8-puzzle.

2.3. Tìm kiếm cục bộ (Local Search)
Các thành phần chính của bài toán tìm kiếm:

Trạng thái ban đầu (Initial state): Điểm xuất phát của bài toán.
Trạng thái đích (Goal state): Mục tiêu cần đạt được.
Hành động (Actions): Các phép biến đổi để di chuyển giữa các trạng thái.
Hàm chi phí (Cost function): Chi phí thực hiện hành động hoặc di chuyển.
Hàm đánh giá (Heuristic function): Ước lượng mức độ gần trạng thái hiện tại đến trạng thái đích.
Solution (Giải pháp): Chuỗi các hành động hoặc trạng thái từ trạng thái ban đầu đến trạng thái đích thỏa mãn yêu cầu bài toán.

Các thuật toán triển khai:

Simple Hill Climbing
Steepest Ascent Hill Climbing
Random Hill Climbing
Simulated Annealing
Beam Search
Genetic Algorithm

Nhận xét:

Các thuật toán Hill Climbing: Nhanh nhưng dễ bị kẹt tại cực trị địa phương, không đảm bảo tìm lời giải tối ưu trong 8-puzzle.
Simulated Annealing: Cải thiện khả năng thoát khỏi cực trị địa phương, phù hợp với bài toán 8-puzzle phức tạp.
Beam Search: Giúp cân bằng giữa bộ nhớ và thời gian, hiệu quả nếu chọn beam width phù hợp.
Genetic Algorithm: Có thể tìm lời giải tốt trong không gian trạng thái lớn nhưng cần nhiều tính toán và tinh chỉnh tham số.
Kết luận: Tìm kiếm cục bộ và tiến hóa cung cấp các phương pháp linh hoạt, có thể áp dụng trong 8-puzzle để tìm lời giải gần tối ưu nhanh hơn so với tìm kiếm toàn diện, nhưng không đảm bảo tối ưu tuyệt đối.

2.4. Tìm kiếm trong môi trường phức tạp (Searching in Complex Environments)
Các thành phần cơ bản của bài toán:

Không gian trạng thái (State space): Tập hợp tất cả các trạng thái có thể xảy ra trong môi trường.
Trạng thái ban đầu (Initial state): Trạng thái xuất phát, nơi quá trình tìm kiếm bắt đầu.
Trạng thái đích hoặc mục tiêu (Goal state): Trạng thái hoặc tập hợp trạng thái mà ta muốn đạt tới.
Toán tử chuyển trạng thái (Actions/Operators): Các phép biến đổi cho phép chuyển từ trạng thái này sang trạng thái khác.
Hàm kiểm tra trạng thái đích (Goal test): Hàm xác định xem trạng thái hiện tại có phải là trạng thái mục tiêu hay không.
Thông tin quan sát (Observability): Trong môi trường phức tạp, có thể trạng thái không được quan sát đầy đủ hoặc chỉ quan sát một phần.
Mô hình môi trường (Model of environment): Mô tả cách trạng thái chuyển đổi dựa trên hành động.
Solution: Chuỗi các hành động hoặc kế hoạch từ trạng thái ban đầu đến trạng thái mục tiêu.

Các thuật toán triển khai:

AND-OR Graph Search
Searching for a partially observation (PartObs)
Sensorless

Nhận xét:

Các thuật toán này phải đối mặt với độ phức tạp rất lớn khi áp dụng cho bài toán 8-puzzle do không gian trạng thái rộng và các hạn chế trong quan sát. Hiệu suất của chúng thường kém và khó áp dụng trực tiếp cho 8-puzzle kích thước đầy đủ mà không có các kỹ thuật tối ưu hóa hoặc giảm không gian trạng thái.

2.5. Bài toán thỏa mãn ràng buộc (Constraint Satisfaction Problems - CSP)
Các thành phần cơ bản:

Tập biến (Variables): Tập hợp các biến cần gán giá trị.
Miền giá trị (Domains): Mỗi biến có một miền giá trị khả dĩ.
Tập ràng buộc (Constraints): Các điều kiện hoặc quan hệ giữa các biến.
Hàm kiểm tra ràng buộc (Constraint Checking): Kiểm tra xem một phép gán giá trị có thỏa mãn tất cả các ràng buộc hay không.
Solution: Một phép gán giá trị cho tất cả các biến sao cho mọi ràng buộc đều được thỏa mãn.

Các thuật toán triển khai:

Backtracking
Forward-Checking

Nhận xét:

Trong nhóm CSP, các thuật toán Backtracking-Search và Forward-Checking có thể áp dụng cho 8-puzzle nhưng hiệu suất thấp hơn nhiều so với các thuật toán heuristic như A*. Forward-Checking chỉ mang lại cải thiện nhỏ so với backtracking thuần túy trong trường hợp này.

2.6. Giới thiệu về học tăng cường (Reinforcement Learning)
Các thành phần cơ bản:

Agent (Tác tử): Là đối tượng ra quyết định, thực hiện các hành động.
Environment (Môi trường): Là bối cảnh mà agent tương tác.
State (Trạng thái): Mô tả tình trạng hiện tại của môi trường.
Action (Hành động): Các lựa chọn mà agent thực hiện.
Reward (Phần thưởng): Phản hồi từ môi trường sau mỗi hành động.
Policy (Chính sách): Quy tắc mà agent sử dụng để chọn hành động.
Value function (Hàm giá trị): Ước lượng tổng phần thưởng kỳ vọng.
Model of Environment: Mô tả cách môi trường chuyển trạng thái và phần thưởng.
Solution: Một chính sách tối ưu, tối đa hóa tổng phần thưởng tích luy.

Các thuật toán triển khai:

Q-Learning

Nhận xét:

Q-Learning truyền thống không phải là phương pháp tối ưu cho bài toán 8-puzzle do không gian trạng thái lớn và phức tạp. Thuật toán này có thể học được chính sách giải nhưng thường rất chậm và tốn nhiều tài nguyên. Các phương pháp kết hợp học sâu hoặc thuật toán heuristic đặc thù thường được ưu tiên hơn trong nhóm trò chơi 8-puzzle.

3. Tác giả

Được phát triển bởi Hồ Miinh Tiến Thành.
