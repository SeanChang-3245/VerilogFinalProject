module draw_once(
    input wire clk,
    input wire rst,
    input wire interboard_rst, 
    input wire draw_one,                    // pass from GameControl, a one-pulse signal that indicate one card should be drawn
    input wire [105:0] available_card, 

    output wire done,
    output wire ready,                      // pass to GameControl, a signal that is always true when the machine is ready to pick another card 
    output reg [6:0] drawn_card_idx         // 0~105, the index in available_card
);

    localparam IDLE = 0;
    localparam TRANSFER = 1;
    localparam CHOOSE = 2;
    localparam FIN = 3;

    reg [3:0] wait_mod_cnt, wait_mod_cnt_next;
    reg [1:0] cur_state, next_state;
    reg [6:0] rnd, rnd_next;
    reg [6:0] drawn_card_idx_next;
    wire [6:0] pick_arr_idx;                // index of array `available_card_idx`, should be in the range [0, available_card_num)
    wire mod_ready, mod_done;
    reg mod_start, mod_start_next;


    integer i;
    reg [6:0] available_card_num;           // record how many cards can be drawn
    reg [6:0] available_card_idx [0:105];   // an array storing all the card "index" that can be drawn (not card type)
    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            i <= 0;
            available_card_num <= 0;
        end
        else if(cur_state == IDLE) begin
            i <= 0;
            available_card_num <= 0;
        end
        else if(cur_state == TRANSFER) begin
            if(available_card[i]) begin
                available_card_idx[available_card_num] <= i;
                available_card_num <= available_card_num + 1;
            end
            i <= i+1;
        end
    end 

    assign done = (cur_state == FIN);
    assign ready = (cur_state == IDLE);
    a_mod_b amb(.clk(clk), .rst(rst), .interboard_rst(rst), 
                .start(mod_start), .ready(mod_ready), .done(mod_done),
                .a(rnd), .b(available_card_num), .ans(pick_arr_idx));
    // assign pick_arr_idx = rnd % available_card_num;
    // assign pick_arr_idx = rnd;

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
            drawn_card_idx <= 110;
            wait_mod_cnt <= 0;
            rnd <= 0;
            mod_start <= 0;
        end 
        else begin
            cur_state <= next_state;
            drawn_card_idx <= drawn_card_idx_next;
            wait_mod_cnt <= wait_mod_cnt_next;
            rnd <= rnd_next;
            mod_start <= mod_start_next;
        end       
    end

    always @(*) begin
        next_state = cur_state;
        if(cur_state == IDLE && draw_one) begin
            next_state = TRANSFER;
        end
        else if(cur_state == TRANSFER && i == 105) begin
            next_state = CHOOSE;
        end
        else if(cur_state == CHOOSE && mod_done) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always@* begin
        if(rnd == 105) begin
            rnd_next = 0;
        end
        else begin
            rnd_next = rnd + 1;
        end
    end

    always@* begin
        drawn_card_idx_next = drawn_card_idx;
        if(cur_state == CHOOSE) begin
            drawn_card_idx_next = available_card_idx[pick_arr_idx];
        end
    end

    always@* begin
        mod_start_next = mod_start;
        if(cur_state == IDLE) begin
            mod_start_next = 0;
        end
        else if(cur_state == CHOOSE) begin
            if(mod_start == 0 && mod_ready) begin
                mod_start_next = 1;
            end
            else begin
                mod_start_next = 0;
            end
        end
    end


endmodule

module a_mod_b(
    input wire clk,
    input wire rst, 
    input wire interboard_rst,
    input wire [6:0] a,
    input wire [6:0] b,
    input wire start,

    output wire ready,
    output wire done,
    output reg [6:0] ans
);

    localparam IDLE = 0;
    localparam CALC = 1;
    localparam FIN = 2;

    reg [1:0] cur_state, next_state;
    reg [6:0] ans_next;
    reg [6:0] stored_b, stored_b_next;      // store the value of b when `start` is triggered

    assign ready = (cur_state == IDLE);
    assign done = (cur_state == FIN);

    always@(posedge clk) begin
        if(rst || interboard_rst) begin
            cur_state <= IDLE;
            ans <= 0;
            stored_b <= 0;
        end
        else begin
            cur_state <= next_state; 
            ans <= ans_next;
            stored_b <= stored_b_next;
        end
    end

    always@* begin
        next_state = cur_state;
        if(cur_state == IDLE && start) begin
            next_state = CALC;
        end
        else if(cur_state == CALC && ans < stored_b) begin
            next_state = FIN;
        end
        else if(cur_state == FIN) begin
            next_state = IDLE;
        end
    end

    always @(*) begin
        ans_next = ans;
        if(cur_state == IDLE && start) begin
            ans_next = a;
        end
        else if(cur_state == CALC && ans >= stored_b) begin
            ans_next = ans - stored_b;
        end
    end

    always@* begin
        stored_b_next = stored_b;
        if(cur_state == IDLE && start) begin
            stored_b_next = b;
        end
    end

endmodule
