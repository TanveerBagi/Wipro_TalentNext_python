def process_purchase_file():
    try:
        filename = input("Enter the purchase file name: ")
        with open(filename, 'r') as f:
            lines = f.readlines()

        total_items = len(lines)
        free_items = 0
        total_amount = 0

        for line in lines:
            try:
                name, price = line.strip().split()
                price = float(price)
                if price == 0:
                    free_items += 1
                total_amount += price
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")

        discount = total_amount * 0.1
        final_amount = total_amount - discount

        print("Total items purchased:", total_items)
        print("Free items:", free_items)
        print("Total amount before discount:", total_amount)
        print("Discount amount:", discount)
        print("Final amount after discount:", final_amount)

    except FileNotFoundError:
        print("File not found. Please check the file name.")
    except Exception as e:
        print("An error occurred:", e)


process_purchase_file()
