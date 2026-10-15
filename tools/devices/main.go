// Command devices lists the devices mimir can use, and why there is no GPU
// if there is none.
//
//	go run ./tools/devices
package main

import (
	"fmt"
	"os"

	"github.com/gsoultan/mimir"
)

func main() {
	for _, d := range mimir.Devices() {
		fmt.Println(d)
	}
	if _, err := mimir.GPU(); err != nil {
		fmt.Fprintln(os.Stderr, err)
	}
}
