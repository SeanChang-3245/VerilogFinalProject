module InterboardCommunication_top(
    input wire clk,
    input wire rst,
    input wire transmit,                       // from game control, indicate it is this board's turn to transmit 
    input wire ctrl_en,
    input wire ctrl_move_dir,
    input wire [4:0] ctrl_block_x,
    input wire [2:0] ctrl_block_y,
    input wire [3:0] ctrl_msg_type,
    input wire [5:0] ctrl_card,
    input wire [2:0] ctrl_sel_len,
    
    inout wire Request,
    inout wire Ack,
    inout wire [5:0] interboard_data,

    output wire interboard_rst,                // rst called from other board
    output wire interboard_en,                 // should be one-pulse
    output wire interboard_move_dir,
    output wire [4:0] interboard_block_x,
    output wire [2:0] interboard_block_y,
    output wire [3:0] interboard_msg_type,
    output wire [5:0] interboard_card,
    output wire [2:0] interboard_sel_len
);
    wire request_in, request_out;
    wire ack_in, ack_out;
    wire [5:0] data_in, data_out;
    
    assign Ack = transmit ? 1'bz : ack_out;
    assign Request = transmit ? request_out : 1'bz;
    assign interboard_data = transmit ? data_out : 6'bzz_zzzz;

    assign ack_in = Ack;                    // ack_in = transmit ? 1'bz(1'b0) : Ack;
    assign request_in = Request;            // request_int = transmit ? 1'bz : Request;
    assign data_in = interboard_data;       // data_in = transmit ? 6'bz : interboard_data;

    
    send_all sa (
        .clk(clk),
        .rst(rst),
        .Ack(ack_in),
        .ctrl_en(ctrl_en),
        .ctrl_move_dir(ctrl_move_dir),
        .ctrl_block_x(ctrl_block_x),
        .ctrl_block_y(ctrl_block_y),
        .ctrl_msg_type(ctrl_msg_type),
        .ctrl_card(ctrl_card),
        .ctrl_sel_len(ctrl_sel_len),

        .Request(request_out),
        .interboard_data(data_out)
    );

    receive_all ra (
        .clk(clk),
        .rst(rst),
        .Request(request_in),
        .interboard_data(data_in),
        
        .Ack(ack_out),
        .interboard_rst(interboard_rst),
        .interboard_en(interboard_en),
        .interboard_move_dir(interboard_move_dir),
        .interboard_block_x(interboard_block_x),
        .interboard_block_y(interboard_block_y),
        .interboard_msg_type(interboard_msg_type),
        .interboard_card(interboard_card),
        .interboard_sel_len(interboard_sel_len)
    );


endmodule



