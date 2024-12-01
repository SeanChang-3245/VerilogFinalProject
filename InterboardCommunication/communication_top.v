module InterboardCommunication_top(
    input wire clk,
    input wire rst,
    input wire ctrl_en,
    input wire ctrl_move_dir,
    input wire [4:0] ctrl_block_x,
    input wire [2:0] ctrl_block_y,
    input wire [3:0] ctrl_msg_type,
    input wire [5:0] ctrl_card,
    input wire [2:0] ctrl_sel_len,
    
    inout wire request,
    inout wire ack,
    inout wire [5:0] interboard_data,

    output wire interboard_rst,
    output wire interboard_en,
    output wire interboard_move_dir,
    output wire [4:0] interboard_block_x,
    output wire [2:0] interboard_block_y,
    output wire [3:0] interboard_msg_type,
    output wire [5:0] interboard_card,
    output wire [2:0] interboard_sel_len
);

    // interboard_rst should be done first 

endmodule