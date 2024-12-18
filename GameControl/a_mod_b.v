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
