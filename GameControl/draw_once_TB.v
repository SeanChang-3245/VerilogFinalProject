module draw_once_TB();

    reg clk, rst;
    initial begin
        clk = 0;
        forever begin
            #5;
            clk = ~clk;
        end
    end

    reg [105:0] available_card, available_card_next;
    reg draw_one;
    wire [6:0] drawn_card_idx;
    wire done, ready;

    draw_once do(.clk(clk), .rst(rst), .interboard_rst(0), .draw_one(draw_one),
                 .available_card(available_card), .done(done), .ready(ready), .drawn_card_idx(drawn_card_idx));
    

    initial begin
        #2000;
        $finish();
    end

    initial begin
        #10;
        rst = 1;
        #10;
        rst = 0;

        #10;

    end

    always @(posedge clk) begin
        if(rst) begin
            available_card <= {106{1'b1}};
        end
        else begin
            available_card <= available_card_next;
        end
    end

    always @(*) begin
        available_card_next = available_card;
        if(done) begin
            available_card_next[drawn_card_idx] = 0;
        end
    end

    always @* begin
        draw_one = ready;
    end

    always @* begin
        if(done) begin
            $display("The card is %d", drawn_card_idx);
        end
    end


endmodule