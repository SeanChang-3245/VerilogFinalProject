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
    
    wire delayed_rst;
    delay_n_cycle #(.n(10)) delay_rst(
        .clk(clk),
        .in(rst),
        .out(delayed_rst)
    );

    // maybe need to handle rst send to other board at top level
    reg rst_other, rst_other_next;
    always@* begin
        rst_other_next = rst_other;
        if(rst) begin
            rst_other_next = 1;
        end
    end

    always@(posedge clk) begin
        if(delayed_rst) begin
            rst_other <= 0;
        end
        else begin
            rst_other <= rst_other_next;
        end
    end
    
    // Data out
    // interboard rst will be triggered if transmit is switcted
    assign Ack = rst_other ? 1 : (transmit ? 1'bz : ack_out);
    assign Request = rst_other ? 1 : (transmit ? request_out : 1'bz);
    assign interboard_data = rst_other ? 1 : (transmit ? data_out : 6'bzz_zzzz);

    // Data in
    assign ack_in = transmit ? Ack : 1'bz;
    assign request_in = transmit ? 1'bz : Request;
    assign data_in = transmit ? 6'bzz_zzzz : interboard_data;

    // rst called by other board
    assign interboard_rst = ({Ack, Request, interboard_data} == 8'hff);

    // assign ack_in = Ack;                    // ack_in = transmit ? 1'bz(1'b0) : Ack;
    // assign request_in = Request;            // request_in = transmit ? 1'bz : Request;
    // assign data_in = interboard_data;       // data_in = transmit ? 6'bz : interboard_data;


    send_all sa (
        .clk(clk),
        .rst(rst),
        .interboard_rst(interboard_rst),
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
        .interboard_rst(interboard_rst),
        .Request(request_in),
        .interboard_data(data_in),
        
        .Ack(ack_out),
        .interboard_en(interboard_en),
        .interboard_move_dir(interboard_move_dir),
        .interboard_block_x(interboard_block_x),
        .interboard_block_y(interboard_block_y),
        .interboard_msg_type(interboard_msg_type),
        .interboard_card(interboard_card),
        .interboard_sel_len(interboard_sel_len)
    );


    ila_1 ila_inst(clk, transmit, Ack, interboard_data, Request, ack_in, request_in, ack_out, request_out);

endmodule



