module draw_place_send_msg_TB();

    reg clk, rst;
    initial begin
        clk = 0;
        forever begin
            #5;
            clk = ~clk;
        end
    end

    reg draw_and_place_en;
    reg inter_ready;
    reg [8*18*6-1:0] map, map_next;
    reg [105:0] available_card, available_card_next;

    wire draw_and_place_done;
    wire draw_and_place_ready;

    wire draw_place_ctrl_en;
    wire draw_place_ctrl_move_dir;
    wire [4:0] draw_place_ctrl_block_x;
    wire [2:0] draw_place_ctrl_block_y;
    wire [3:0] draw_place_ctrl_msg_type;
    wire [5:0] draw_place_ctrl_card;
    wire [2:0] draw_place_ctrl_sel_len;


    draw_one_place_send_msg draw_place_inst(
        .clk(clk),
        .rst(rst), 
        .interboard_rst(0),

        .draw_and_place_en(draw_and_place_en),
        .inter_ready(inter_ready),
        .map(map),
        .available_card(available_card),

        .draw_and_place_done(draw_and_place_done),
        .draw_and_place_ready(draw_and_place_ready),

        .draw_place_ctrl_en(draw_place_ctrl_en),
        .draw_place_ctrl_move_dir(draw_place_ctrl_move_dir),
        .draw_place_ctrl_block_x(draw_place_ctrl_block_x),
        .draw_place_ctrl_block_y(draw_place_ctrl_block_y),
        .draw_place_ctrl_sel_len(draw_place_ctrl_sel_len),
        .draw_place_ctrl_card(draw_place_ctrl_card),
        .draw_place_ctrl_msg_type(draw_place_ctrl_msg_type)
    );

    initial begin
        available_card = {{100{1'b0}}, 6'b111111};
        map = {{34{6'd54}}, 6'd0, 6'd0, {108{6'b0}}};
        inter_ready = 1;
    end

    // initial begin
    //     wait(draw_place_ctrl_en == 1);
    //     #55;
    //     inter_ready = 1;
    //     #10; 
    //     inter_ready = 0;

    //     wait(draw_place_ctrl_en == 1);
    //     #60;
    //     inter_ready = 1;
    //     #10; 
    //     inter_ready = 0;
    // end

    always@* begin
        available_card_next = available_card;
        map_next = map;
        if(draw_place_ctrl_en) begin
            available_card_next = available_card ^ (1 << draw_place_ctrl_card);
            map_next[draw_place_ctrl_block_x*6 + draw_place_ctrl_block_y*6*18 +: 6] = draw_place_ctrl_card;
        end
    end



    initial begin
        #200;
        $finish();
    end

    initial begin
        #10;
        rst = 1;
        #10;
        rst = 0;

        #10;
        draw_and_place_en = 1;
        #10;
        draw_and_place_en = 0;

        wait(draw_and_place_done == 1);
        #55;
        draw_and_place_en = 1;
        #10;
        draw_and_place_en = 0;

        wait(draw_and_place_done == 1);
        #55;
        draw_and_place_en = 1;
        #10;
        draw_and_place_en = 0;

    end

    always@(posedge clk) begin
        if(rst) begin
            available_card <= {{100{1'b0}}, 6'b111111};
            map <= {{34{6'd54}}, 6'd0, 6'd0, {108{6'b0}}};
        end
        else begin
            available_card <= available_card_next;
            map <= map_next;
        end
    end
    


    always @(*) begin
        if(draw_place_ctrl_en) begin
            $display("==================================================");
            $display("draw_place_ctrl_en = %d", draw_place_ctrl_en);
            $display("draw_place_ctrl_move_dir = %d", draw_place_ctrl_move_dir);
            $display("draw_place_ctrl_block_x = %d", draw_place_ctrl_block_x);
            $display("draw_place_ctrl_block_y = %d", draw_place_ctrl_block_y);
            $display("draw_place_ctrl_msg_type = %d", draw_place_ctrl_msg_type);
            $display("draw_place_ctrl_card = %d", draw_place_ctrl_card);
            $display("draw_place_ctrl_sel_len = %d", draw_place_ctrl_sel_len);
            $display("==================================================\n\n");
        end
    end

    


endmodule