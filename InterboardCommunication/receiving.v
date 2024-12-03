module receive_all(
    input wire clk,
    input wire rst,
    input wire Request,
    input wire [5:0] interboard_data,

    output wire Ack,
    output wire interboard_rst,
    output wire interboard_en,                  // to upper layer, one-pulse
    output wire [3:0] interboard_msg_type,
    output wire [4:0] interboard_block_x,
    output wire [2:0] interboard_block_y,
    output wire [5:0] interboard_card,
    output wire [2:0] interboard_sel_len,
    output wire interboard_move_dir
);

    // Transmission state
    localparam WAIT_1 = 0; // msg_type
    localparam WAIT_2 = 1; // block_x 
    localparam WAIT_3 = 2; // block_y
    localparam WAIT_4 = 3; // card
    localparam WAIT_5 = 4; // sel_len
    localparam WAIT_6 = 5; // move_dir
    localparam FININSH = 6;

    // Used to store data to be sent, passed from other board
    reg [3:0] stored_msg_type, stored_msg_type_next;
    reg [4:0] stored_block_x, stored_block_x_next;
    reg [2:0] stored_block_y, stored_block_y_next;
    reg [5:0] stored_card, stored_card_next;
    reg [2:0] stored_sel_len, stored_sel_len_next;
    reg stored_move_dir, stored_move_dir_next;

    reg [2:0] cur_state, next_state;
    wire [5:0] cur_data;
    wire done;

    single_receive sr(
        .clk(clk),
        .rst(rst),
        .Request(Request),
        .interboard_data(interboard_data),

        .done(done),
        .Ack(Ack),
        .data_out(cur_data)
    );

    always@(posedge clk) begin
        if(rst) begin
            cur_state <= WAIT_1;
            stored_msg_type <= 0;
            stored_block_x <= 0;
            stored_block_y <= 0;
            stored_card <= 0;
            stored_sel_len <= 0;
            stored_move_dir <= 0;
        end
        else begin
            cur_state <= next_state;
            stored_msg_type <= stored_msg_type_next;
            stored_block_x <= stored_block_x_next;
            stored_block_y <= stored_block_y_next;
            stored_card <= stored_card_next;
            stored_sel_len <= stored_sel_len_next;
            stored_move_dir <= stored_move_dir_next;
        end
    end

    assign interboard_en = (cur_state == FININSH);
    assign interboard_msg_type = stored_msg_type;
    assign interboard_block_x = stored_block_x;
    assign interboard_block_y = stored_block_y;
    assign interboard_card = stored_card;
    assign interboard_sel_len = stored_sel_len;
    assign interboard_move_dir = stored_move_dir;
    assign interboard_rst = {Request, interboard_data} == 7'b111_1111;

    always@* begin
        next_state = cur_state;
        if(cur_state == WAIT_1 && done) begin
            next_state = WAIT_2;
        end
        else if(cur_state == WAIT_2 && done) begin
            next_state = WAIT_3;
        end
        else if(cur_state == WAIT_3 && done) begin
            next_state = WAIT_4;
        end
        else if(cur_state == WAIT_4 && done) begin
            next_state = WAIT_5;
        end
        else if(cur_state == WAIT_5 && done) begin
            next_state = WAIT_6;
        end
        else if(cur_state == WAIT_6 && done) begin
            next_state = FININSH;
        end
        else if(cur_state == FININSH) begin
            next_state = WAIT_1;
        end
    end

    always@* begin
        stored_msg_type_next = stored_msg_type;
        stored_block_x_next = stored_block_x;
        stored_block_y_next = stored_block_y;
        stored_card_next = stored_card;
        stored_sel_len_next = stored_sel_len;
        stored_move_dir_next = stored_move_dir;
        if(cur_state == WAIT_1) begin
            stored_msg_type_next = cur_data;
        end
        else if(cur_state == WAIT_2) begin
            stored_block_x_next = cur_data;
        end
        else if(cur_state == WAIT_3) begin
            stored_block_y_next = cur_data;
        end
        else if(cur_state == WAIT_4) begin
            stored_card_next = cur_data;
        end
        else if(cur_state == WAIT_5) begin
            stored_sel_len_next = cur_data;
        end
        else if(cur_state == WAIT_6) begin
            stored_move_dir_next = cur_data;
        end
    end
endmodule



module single_receive(
    input wire clk,
    input wire rst, 
    input wire Request,
    input wire [5:0] interboard_data,   // from other board
  
    output wire done,                   // to upper layer, indicate one round is end, the data should be retrieved, one-pulse
    output wire Ack,
    output wire [5:0] data_out          // to upper layer
);
    localparam ACK_TIME = 10;
  
    localparam WAIT_REQ = 0;
    localparam ACK_STATE = 1;

    reg cur_state, next_state;
    reg [9:0] counter, counter_next;

    assign done = (cur_state == ACK_STATE && counter == ACK_STATE);

    always@(posedge clk) begin
        if(rst) begin
            cur_state <= WAIT_REQ;
            counter <= 0;
        end 
        else begin
            cur_state <= next_state;
            counter <= counter_next;
        end   
    end

    always@* begin
        next_state = cur_state;
        if(cur_state == WAIT_REQ && Request) begin
            next_state = ACK_STATE;
        end
        else if(cur_state == ACK_STATE && counter == ACK_TIME) begin
            next_state = WAIT_REQ;
        end
    end

    assign Ack = (cur_state == ACK_STATE);
    assign data_out = interboard_data;

    always@(*) begin
        counter_next = counter;
        if(cur_state == WAIT_REQ) begin
            counter_next = 0;
        end
        else if(cur_state == ACK_STATE) begin
            counter_next = counter + 1;
        end
    end


endmodule